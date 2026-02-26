"""Fast experiment runner optimized for CPU.

Reduced data sizes but still enough for meaningful signal.
Uses teacher-forced digit accuracy as proxy and only does
autoregressive eval occasionally.
"""

import argparse
import math
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset

# ── Constants ──
VOCAB_SIZE = 14
D_MODEL = 6
TOK_DIM = 3
POS_DIM = 3
N_HEADS = 2
HEAD_DIM = 3
FFN_DIM = 6
MAX_SEQ_LEN = 34
MAX_DIGITS = 10
ANSWER_LEN = 11
N_POS = 10

PLUS_TOKEN = 10
EQUALS_TOKEN = 11
EOS_TOKEN = 12
PAD_TOKEN = 13

EQ_POS = 2 * MAX_DIGITS + 1  # 21
X_START = 0
PLUS_POS = MAX_DIGITS
Y_START = MAX_DIGITS + 1
Z_START = 2 * MAX_DIGITS + 2
EOS_POS = MAX_SEQ_LEN - 1

# Fast training settings
BATCH_SIZE = 512
LR = 1e-3
EPOCHS = 200
TRAIN_SAMPLES = 10000  # per epoch
VAL_SIZE = 1000
PATIENCE = 40  # in terms of eval intervals
EVAL_EVERY = 3  # eval every N epochs
FULL_EVAL_EVERY = 15  # autoregressive eval every N epochs


def _spiral(index, period):
    return (
        math.cos(2 * math.pi * index / period),
        math.sin(2 * math.pi * index / period),
        index / max(period - 1, 1),
    )


def build_token_embedding():
    emb = torch.zeros(VOCAB_SIZE, TOK_DIM)
    for d in range(10):
        emb[d, 0], emb[d, 1], emb[d, 2] = _spiral(d, 10)
    emb[PLUS_TOKEN]   = torch.tensor([2.0, 0.0, -1.0])
    emb[EQUALS_TOKEN] = torch.tensor([0.0, 2.0, -1.0])
    emb[EOS_TOKEN]    = torch.tensor([-2.0, 0.0, -1.0])
    emb[PAD_TOKEN]    = torch.tensor([0.0, -2.0, -1.0])
    return emb


def build_positional_encoding():
    pe = torch.zeros(MAX_SEQ_LEN, POS_DIM)
    for i in range(MAX_DIGITS):
        pe[X_START + i, 0], pe[X_START + i, 1], pe[X_START + i, 2] = _spiral(i, N_POS)
    for i in range(MAX_DIGITS):
        pe[Y_START + i, 0], pe[Y_START + i, 1], pe[Y_START + i, 2] = _spiral(i, N_POS)
    for i in range(min(ANSWER_LEN, N_POS)):
        pe[Z_START + i, 0], pe[Z_START + i, 1], pe[Z_START + i, 2] = _spiral(i, N_POS)
    pe[Z_START + 10] = torch.tensor([0.0, 0.0, 1.5])
    pe[PLUS_POS] = torch.tensor([2.0, 0.0, -1.0])
    pe[EQ_POS]   = torch.tensor([0.0, 2.0, -1.0])
    pe[EOS_POS]  = torch.tensor([-2.0, 0.0, -1.0])
    return pe


# ── Data ──
_PROMPT_LEN = MAX_DIGITS + 1 + MAX_DIGITS + 1  # 22
_ANSWER_LEN_DATA = MAX_DIGITS + 1 + 1  # 12
_LOSS_MASK = np.array([0] * _PROMPT_LEN + [1] * _ANSWER_LEN_DATA, dtype=np.int64)


def generate_batch_np(num_samples, rng, digits_only=None):
    if digits_only is not None:
        n = np.full(num_samples, digits_only, dtype=np.int64)
    else:
        n = rng.integers(1, MAX_DIGITS + 1, size=num_samples)
    lo = np.where(n == 1, 0, 10 ** (n - 1))
    hi = 10 ** n
    x = (rng.random(num_samples) * (hi - lo) + lo).astype(np.int64)
    y = (rng.random(num_samples) * (hi - lo) + lo).astype(np.int64)
    z = x + y
    tokens = np.empty((num_samples, MAX_SEQ_LEN), dtype=np.int64)
    tmp = x.copy()
    for d in range(MAX_DIGITS):
        tokens[:, d] = tmp % 10
        tmp //= 10
    tokens[:, MAX_DIGITS] = PLUS_TOKEN
    tmp = y.copy()
    for d in range(MAX_DIGITS):
        tokens[:, MAX_DIGITS + 1 + d] = tmp % 10
        tmp //= 10
    tokens[:, 2 * MAX_DIGITS + 1] = EQUALS_TOKEN
    tmp = z.copy()
    for d in range(MAX_DIGITS + 1):
        tokens[:, 2 * MAX_DIGITS + 2 + d] = tmp % 10
        tmp //= 10
    tokens[:, -1] = EOS_TOKEN
    return tokens, n


class AdditionDataset(Dataset):
    def __init__(self, num_samples, seed=None, digits_only=None):
        rng = np.random.default_rng(seed)
        tokens_np, n_digits_np = generate_batch_np(num_samples, rng, digits_only=digits_only)
        self.tokens = torch.from_numpy(tokens_np)
        self.loss_mask = torch.from_numpy(np.tile(_LOSS_MASK, (num_samples, 1)))
        self.n_digits = torch.from_numpy(n_digits_np)

    def __len__(self):
        return self.tokens.shape[0]

    def __getitem__(self, idx):
        return self.tokens[idx], self.loss_mask[idx], self.n_digits[idx]


# ── Attention Variants ──

class QKPosVTokAttention(nn.Module):
    def __init__(self, n_heads=N_HEADS):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = HEAD_DIM
        self.q_proj = nn.Linear(POS_DIM, n_heads * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(POS_DIM, n_heads * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(TOK_DIM, n_heads * HEAD_DIM, bias=False)
        self.out_proj = nn.Linear(n_heads * HEAD_DIM, D_MODEL, bias=False)
        self.register_buffer("causal_mask",
            torch.tril(torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN)).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        B, T, _ = x.shape
        x_pos = x[:, :, TOK_DIM:]
        x_tok = x[:, :, :TOK_DIM]
        q = self.q_proj(x_pos).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_pos).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_tok).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.out_proj(out)


class SharedQKAttention(nn.Module):
    def __init__(self, n_heads=N_HEADS):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = HEAD_DIM
        self.qk_proj = nn.Linear(POS_DIM, n_heads * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(TOK_DIM, n_heads * HEAD_DIM, bias=False)
        self.out_proj = nn.Linear(n_heads * HEAD_DIM, D_MODEL, bias=False)
        self.register_buffer("causal_mask",
            torch.tril(torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN)).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        B, T, _ = x.shape
        x_pos = x[:, :, TOK_DIM:]
        x_tok = x[:, :, :TOK_DIM]
        qk = self.qk_proj(x_pos).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_tok).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        attn = (qk @ qk.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.out_proj(out)


class StandardAttention(nn.Module):
    def __init__(self, n_heads=N_HEADS):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = D_MODEL // n_heads
        self.q_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.k_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.v_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.out_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.register_buffer("causal_mask",
            torch.tril(torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN)).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D_MODEL)
        return self.out_proj(out)


class SplitHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = HEAD_DIM
        self.q1 = nn.Linear(TOK_DIM, HEAD_DIM, bias=False)
        self.k1 = nn.Linear(TOK_DIM, HEAD_DIM, bias=False)
        self.v1 = nn.Linear(TOK_DIM, HEAD_DIM, bias=False)
        self.q2 = nn.Linear(POS_DIM, HEAD_DIM, bias=False)
        self.k2 = nn.Linear(POS_DIM, HEAD_DIM, bias=False)
        self.v2 = nn.Linear(POS_DIM, HEAD_DIM, bias=False)
        self.out_proj = nn.Linear(N_HEADS * HEAD_DIM, D_MODEL, bias=False)
        self.register_buffer("causal_mask",
            torch.tril(torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN)).unsqueeze(0))

    def _attend(self, q, k, v, T):
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(self.causal_mask[:, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        return attn @ v

    def forward(self, x):
        B, T, _ = x.shape
        x_tok = x[:, :, :TOK_DIM]
        x_pos = x[:, :, TOK_DIM:]
        out1 = self._attend(self.q1(x_tok), self.k1(x_tok), self.v1(x_tok), T)
        out2 = self._attend(self.q2(x_pos), self.k2(x_pos), self.v2(x_pos), T)
        out = torch.cat([out1, out2], dim=-1)
        return self.out_proj(out)


# ── Blocks ──

class BlockWithFFN(nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = attn
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.ffn = nn.Sequential(
            nn.Linear(D_MODEL, FFN_DIM),
            nn.GELU(),
            nn.Linear(FFN_DIM, D_MODEL),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class BlockNoFFN(nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.ln = nn.LayerNorm(D_MODEL)
        self.attn = attn

    def forward(self, x):
        return x + self.attn(self.ln(x))


# ── Configurable Model ──

class Model(nn.Module):
    def __init__(self, freeze_tok=False, freeze_pos=False, use_ffn=True,
                 attn_type='qkpos_vtok', n_layers=1, n_heads=N_HEADS):
        super().__init__()
        if freeze_tok:
            self.register_buffer("tok_emb_buf", build_token_embedding())
            self._tok_mode = 'buffer'
        else:
            self.tok_emb = nn.Embedding(VOCAB_SIZE, TOK_DIM)
            self.tok_emb.weight.data.copy_(build_token_embedding())
            self._tok_mode = 'param'

        if freeze_pos:
            self.register_buffer("pos_enc_buf", build_positional_encoding())
            self._pos_mode = 'buffer'
        else:
            self.pos_enc = nn.Parameter(build_positional_encoding())
            self._pos_mode = 'param'

        blocks = []
        for _ in range(n_layers):
            if attn_type == 'qkpos_vtok':
                attn = QKPosVTokAttention(n_heads)
            elif attn_type == 'shared_qk':
                attn = SharedQKAttention(n_heads)
            elif attn_type == 'split_heads':
                attn = SplitHeadAttention()
            elif attn_type == 'standard':
                attn = StandardAttention(n_heads)
            else:
                raise ValueError(f"Unknown: {attn_type}")
            blocks.append(BlockWithFFN(attn) if use_ffn else BlockNoFFN(attn))

        self.blocks = nn.ModuleList(blocks)
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if name in ('tok_emb.weight', 'pos_enc'):
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, idx):
        B, T = idx.shape
        tok = self.tok_emb_buf[idx] if self._tok_mode == 'buffer' else self.tok_emb(idx)
        pos = (self.pos_enc_buf if self._pos_mode == 'buffer' else self.pos_enc)[:T].unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([tok, pos], dim=-1)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)

    @torch.no_grad()
    def generate(self, prompt):
        self.eval()
        B, T_prompt = prompt.shape
        full_seq = torch.zeros(B, T_prompt + ANSWER_LEN + 1, dtype=torch.long, device=prompt.device)
        full_seq[:, :T_prompt] = prompt
        for step in range(ANSWER_LEN + 1):
            T = T_prompt + step
            logits = self.forward(full_seq[:, :T])
            full_seq[:, T] = logits[:, -1].argmax(dim=-1)
        return full_seq[:, T_prompt:]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad), \
               sum(p.numel() for p in self.parameters())


def fast_eval(model, dataloader, do_autoreg=False):
    """Fast evaluation. Only does autoregressive if do_autoreg=True."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    total_exact = 0
    total_sequences = 0

    with torch.no_grad():
        for tokens, loss_mask, n_digits in dataloader:
            logits = model(tokens[:, :-1])
            targets = tokens[:, 1:]
            mask = loss_mask[:, 1:]

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction='none'
            ).reshape(targets.shape)

            masked_loss = (loss * mask).sum()
            n_tok = mask.sum()
            if n_tok > 0:
                total_loss += masked_loss.item()
                total_tokens += n_tok.item()

            preds_tf = logits.argmax(dim=-1)
            correct = (preds_tf == targets) & (mask == 1)
            total_correct += correct.sum().item()
            total_sequences += tokens.size(0)

            if do_autoreg:
                prompt = tokens[:, :EQ_POS + 1]
                answer_target = tokens[:, EQ_POS + 1:]
                generated = model.generate(prompt)
                exact = (generated == answer_target).all(dim=-1)
                total_exact += exact.sum().item()

    avg_loss = total_loss / max(total_tokens, 1)
    digit_acc = total_correct / max(total_tokens, 1)
    exact_acc = total_exact / max(total_sequences, 1) if do_autoreg else None
    return avg_loss, digit_acc, exact_acc


# ── Experiments ──

EXPERIMENTS = {
    'baseline': dict(freeze_tok=False, freeze_pos=False, use_ffn=True, attn_type='qkpos_vtok', n_layers=1, n_heads=2, digits_only=None),
    'freeze_tok': dict(freeze_tok=True, freeze_pos=False, use_ffn=True, attn_type='qkpos_vtok', n_layers=1, n_heads=2, digits_only=None),
    'freeze_pos': dict(freeze_tok=False, freeze_pos=True, use_ffn=True, attn_type='qkpos_vtok', n_layers=1, n_heads=2, digits_only=None),
    'freeze_both': dict(freeze_tok=True, freeze_pos=True, use_ffn=True, attn_type='qkpos_vtok', n_layers=1, n_heads=2, digits_only=None),
    'no_ffn': dict(freeze_tok=False, freeze_pos=False, use_ffn=False, attn_type='qkpos_vtok', n_layers=1, n_heads=2, digits_only=None),
    'no_ffn_freeze_both': dict(freeze_tok=True, freeze_pos=True, use_ffn=False, attn_type='qkpos_vtok', n_layers=1, n_heads=2, digits_only=None),
    'no_ffn_2L': dict(freeze_tok=False, freeze_pos=False, use_ffn=False, attn_type='qkpos_vtok', n_layers=2, n_heads=2, digits_only=None),
    'digits10_only': dict(freeze_tok=False, freeze_pos=False, use_ffn=True, attn_type='qkpos_vtok', n_layers=1, n_heads=2, digits_only=10),
    'standard_attn': dict(freeze_tok=False, freeze_pos=False, use_ffn=True, attn_type='standard', n_layers=1, n_heads=2, digits_only=None),
    'split_heads': dict(freeze_tok=False, freeze_pos=False, use_ffn=True, attn_type='split_heads', n_layers=1, n_heads=2, digits_only=None),
    '1head': dict(freeze_tok=False, freeze_pos=False, use_ffn=True, attn_type='qkpos_vtok', n_layers=1, n_heads=1, digits_only=None),
    'shared_qk': dict(freeze_tok=False, freeze_pos=False, use_ffn=True, attn_type='shared_qk', n_layers=1, n_heads=2, digits_only=None),
    'shared_qk_no_ffn': dict(freeze_tok=False, freeze_pos=False, use_ffn=False, attn_type='shared_qk', n_layers=1, n_heads=2, digits_only=None),
    'minimal': dict(freeze_tok=True, freeze_pos=True, use_ffn=False, attn_type='shared_qk', n_layers=1, n_heads=2, digits_only=None),
}


def run_one(name, config):
    digits_only = config.pop('digits_only', None)
    model = Model(**config)
    n_train, n_total = model.count_parameters()

    print(f"\n{'='*60}")
    print(f"EXP: {name} | {n_train} trainable / {n_total} counted params")
    print(f"{'='*60}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    val_ds = AdditionDataset(VAL_SIZE, seed=42)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    best_digit_acc = 0
    best_exact = 0
    patience = 0
    start = time.time()
    epoch_at_99 = None

    for epoch in range(1, EPOCHS + 1):
        train_ds = AdditionDataset(TRAIN_SAMPLES, seed=None, digits_only=digits_only)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

        model.train()
        ep_loss = 0
        ep_tok = 0
        for tokens, loss_mask, _ in train_loader:
            logits = model(tokens[:, :-1])
            targets = tokens[:, 1:]
            mask = loss_mask[:, 1:]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction='none'
            ).reshape(targets.shape)
            ml = (loss * mask).sum()
            nt = mask.sum()
            if nt > 0:
                (ml / nt).backward()
                optimizer.step()
                optimizer.zero_grad()
                ep_loss += ml.item()
                ep_tok += nt.item()

        # Quick eval every EVAL_EVERY epochs
        if epoch % EVAL_EVERY == 0 or epoch <= 5:
            do_ar = (epoch % FULL_EVAL_EVERY == 0) or (best_digit_acc > 0.95)
            vl, da, ea = fast_eval(model, val_loader, do_autoreg=do_ar)
            elapsed = time.time() - start

            status = f"E{epoch:3d} [{elapsed:5.0f}s] loss={ep_loss/max(ep_tok,1):.4f} digit={da:.4f}"
            if ea is not None:
                status += f" exact={ea:.4f}"
                if ea > best_exact:
                    best_exact = ea
                    torch.save(model.state_dict(), f'exp_{name}_best.pt')
                if epoch_at_99 is None and ea >= 0.99:
                    epoch_at_99 = epoch
                    status += " *** 99%! ***"
            print(status, flush=True)

            if da > best_digit_acc:
                best_digit_acc = da
                patience = 0
                if not do_ar:
                    torch.save(model.state_dict(), f'exp_{name}_best.pt')
            else:
                patience += 1

            if patience >= PATIENCE:
                print(f"  Early stop at epoch {epoch}")
                break

    total_time = time.time() - start

    # Final big eval
    print(f"\n--- Final eval ({name}) ---")
    if os.path.exists(f'exp_{name}_best.pt'):
        model.load_state_dict(torch.load(f'exp_{name}_best.pt', weights_only=True))
    test_ds = AdditionDataset(5000, seed=999)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
    _, _, final_exact = fast_eval(model, test_loader, do_autoreg=True)

    print(f"  Params: {n_total} | Final exact: {final_exact:.4f} | Best exact: {best_exact:.4f} | "
          f"Epoch@99%: {epoch_at_99} | Time: {total_time:.0f}s")

    return {
        'name': name, 'params': n_total, 'final_exact': final_exact,
        'best_exact': best_exact, 'epoch_at_99': epoch_at_99,
        'best_digit_acc': best_digit_acc, 'time': total_time,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', nargs='+', default=['baseline'],
                        help='Experiment name(s) or "all"')
    args = parser.parse_args()

    if 'all' in args.exp:
        names = list(EXPERIMENTS.keys())
    elif 'freeze_ablation' in args.exp:
        names = ['baseline', 'freeze_tok', 'freeze_pos', 'freeze_both']
    elif 'quick' in args.exp:
        names = ['baseline', 'freeze_both', 'no_ffn', 'no_ffn_freeze_both', 'minimal']
    else:
        names = args.exp

    results = []
    for name in names:
        if name not in EXPERIMENTS:
            print(f"Unknown experiment: {name}")
            continue
        cfg = EXPERIMENTS[name].copy()
        r = run_one(name, cfg)
        results.append(r)

    # Summary
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Name':<30} {'Params':>6} {'Final':>7} {'Best':>7} {'Ep@99%':>7} {'Time':>6}")
    print('-' * 70)
    for r in results:
        print(f"{r['name']:<30} {r['params']:>6} {r['final_exact']:>7.4f} "
              f"{r['best_exact']:>7.4f} {str(r['epoch_at_99']):>7} {r['time']:>5.0f}s")


if __name__ == '__main__':
    main()
