"""Unified experiment runner for ablation studies.

Usage:
    python experiment.py --exp freeze_tok     # Freeze token embedding
    python experiment.py --exp freeze_pos     # Freeze positional encoding
    python experiment.py --exp freeze_both    # Freeze both embeddings
    python experiment.py --exp no_ffn         # Remove FFN
    python experiment.py --exp no_ffn_freeze  # No FFN + freeze both
    python experiment.py --exp digits10_only  # Train on 10-digit only
    python experiment.py --exp baseline       # Reproduce baseline (trainable emb)
    python experiment.py --exp standard_attn  # Standard attention (QKV from full d_model)
    python experiment.py --exp split_heads    # Split heads (head1=tok, head2=pos)
    python experiment.py --exp 1head          # Single head
    python experiment.py --exp no_ffn_1head   # No FFN + 1 head
    python experiment.py --exp shared_qk      # Shared Q=K projection
"""

import argparse
import math
import os
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

X_START = 0
PLUS_POS = MAX_DIGITS  # 10
Y_START = MAX_DIGITS + 1  # 11
EQ_POS = 2 * MAX_DIGITS + 1  # 21
Z_START = 2 * MAX_DIGITS + 2  # 22
EOS_POS = MAX_SEQ_LEN - 1  # 33

# Training defaults
BATCH_SIZE = 256
LR = 1e-3
EPOCHS = 300  # Reduced for ablation (we know it converges in ~75)
VAL_SIZE = 2048
TRAIN_SAMPLES_PER_EPOCH = 50000
LOG_EVERY = 50
PATIENCE = 100


# ── Spiral helpers ──
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
    """Generate addition examples. If digits_only is set, use that many digits for all samples."""
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
    """Q,K from positional dims, V from token dims."""
    def __init__(self, n_heads=N_HEADS):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = HEAD_DIM
        self.q_proj = nn.Linear(POS_DIM, n_heads * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(POS_DIM, n_heads * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(TOK_DIM, n_heads * HEAD_DIM, bias=False)
        self.out_proj = nn.Linear(n_heads * HEAD_DIM, D_MODEL, bias=False)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN)).unsqueeze(0).unsqueeze(0)
        )

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
    """Shared Q=K projection from positional dims, V from token dims."""
    def __init__(self, n_heads=N_HEADS):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = HEAD_DIM
        self.qk_proj = nn.Linear(POS_DIM, n_heads * HEAD_DIM, bias=False)  # shared
        self.v_proj = nn.Linear(TOK_DIM, n_heads * HEAD_DIM, bias=False)
        self.out_proj = nn.Linear(n_heads * HEAD_DIM, D_MODEL, bias=False)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN)).unsqueeze(0).unsqueeze(0)
        )

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


class SplitHeadAttention(nn.Module):
    """Head 1: QKV from tok dims. Head 2: QKV from pos dims."""
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
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN)).unsqueeze(0)
        )

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


class StandardAttention(nn.Module):
    """Standard multi-head attention from full d_model."""
    def __init__(self, n_heads=N_HEADS):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = D_MODEL // n_heads
        self.q_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.k_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.v_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.out_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN)).unsqueeze(0).unsqueeze(0)
        )

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


# ── Transformer Blocks ──

class TransformerBlockWithFFN(nn.Module):
    def __init__(self, attn_module):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = attn_module
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


class TransformerBlockNoFFN(nn.Module):
    def __init__(self, attn_module):
        super().__init__()
        self.ln = nn.LayerNorm(D_MODEL)
        self.attn = attn_module

    def forward(self, x):
        return x + self.attn(self.ln(x))


# ── Configurable Model ──

class ConfigurableTransformer(nn.Module):
    def __init__(self, freeze_tok=False, freeze_pos=False, use_ffn=True,
                 attn_type='qkpos_vtok', n_layers=1, n_heads=N_HEADS):
        super().__init__()

        # Embeddings
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

        # Build attention + blocks
        blocks = []
        for _ in range(n_layers):
            if attn_type == 'qkpos_vtok':
                attn = QKPosVTokAttention(n_heads=n_heads)
            elif attn_type == 'shared_qk':
                attn = SharedQKAttention(n_heads=n_heads)
            elif attn_type == 'split_heads':
                attn = SplitHeadAttention()
            elif attn_type == 'standard':
                attn = StandardAttention(n_heads=n_heads)
            else:
                raise ValueError(f"Unknown attn_type: {attn_type}")

            if use_ffn:
                blocks.append(TransformerBlockWithFFN(attn))
            else:
                blocks.append(TransformerBlockNoFFN(attn))

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

    def _get_tok(self, idx):
        if self._tok_mode == 'buffer':
            return self.tok_emb_buf[idx]
        else:
            return self.tok_emb(idx)

    def _get_pos(self, T, B):
        if self._pos_mode == 'buffer':
            return self.pos_enc_buf[:T].unsqueeze(0).expand(B, -1, -1)
        else:
            return self.pos_enc[:T].unsqueeze(0).expand(B, -1, -1)

    def forward(self, idx):
        B, T = idx.shape
        tok = self._get_tok(idx)
        pos = self._get_pos(T, B)
        x = torch.cat([tok, pos], dim=-1)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)

    @torch.no_grad()
    def generate(self, prompt):
        self.eval()
        idx = prompt
        generated = []
        for _ in range(ANSWER_LEN + 1):
            logits = self.forward(idx)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            generated.append(next_token)
            idx = torch.cat([idx, next_token], dim=1)
        return torch.cat(generated, dim=1)

    def count_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total


# ── Evaluation ──

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    total_exact = 0
    total_sequences = 0
    bucket_correct = defaultdict(int)
    bucket_total = defaultdict(int)

    with torch.no_grad():
        for tokens, loss_mask, n_digits in dataloader:
            tokens = tokens.to(device)
            loss_mask = loss_mask.to(device)

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

            prompt = tokens[:, :EQ_POS + 1]
            answer_target = tokens[:, EQ_POS + 1:]
            generated = model.generate(prompt)

            exact = (generated == answer_target).all(dim=-1)
            total_exact += exact.sum().item()
            total_sequences += tokens.size(0)

            for nd in range(1, MAX_DIGITS + 1):
                mask_nd = (n_digits == nd)
                if mask_nd.any():
                    bucket_total[nd] += mask_nd.sum().item()
                    bucket_correct[nd] += (exact & mask_nd.to(device)).sum().item()

    avg_loss = total_loss / max(total_tokens, 1)
    digit_acc = total_correct / max(total_tokens, 1)
    exact_acc = total_exact / max(total_sequences, 1)
    bucket_acc = {
        nd: bucket_correct[nd] / max(bucket_total[nd], 1)
        for nd in sorted(bucket_total.keys())
    }
    return avg_loss, digit_acc, exact_acc, bucket_acc


# ── Experiment configs ──

EXPERIMENTS = {
    'baseline': dict(
        freeze_tok=False, freeze_pos=False, use_ffn=True,
        attn_type='qkpos_vtok', n_layers=1, n_heads=2,
        digits_only=None,
    ),
    'freeze_tok': dict(
        freeze_tok=True, freeze_pos=False, use_ffn=True,
        attn_type='qkpos_vtok', n_layers=1, n_heads=2,
        digits_only=None,
    ),
    'freeze_pos': dict(
        freeze_tok=False, freeze_pos=True, use_ffn=True,
        attn_type='qkpos_vtok', n_layers=1, n_heads=2,
        digits_only=None,
    ),
    'freeze_both': dict(
        freeze_tok=True, freeze_pos=True, use_ffn=True,
        attn_type='qkpos_vtok', n_layers=1, n_heads=2,
        digits_only=None,
    ),
    'no_ffn': dict(
        freeze_tok=False, freeze_pos=False, use_ffn=False,
        attn_type='qkpos_vtok', n_layers=1, n_heads=2,
        digits_only=None,
    ),
    'no_ffn_freeze_both': dict(
        freeze_tok=True, freeze_pos=True, use_ffn=False,
        attn_type='qkpos_vtok', n_layers=1, n_heads=2,
        digits_only=None,
    ),
    'no_ffn_2layer': dict(
        freeze_tok=False, freeze_pos=False, use_ffn=False,
        attn_type='qkpos_vtok', n_layers=2, n_heads=2,
        digits_only=None,
    ),
    'digits10_only': dict(
        freeze_tok=False, freeze_pos=False, use_ffn=True,
        attn_type='qkpos_vtok', n_layers=1, n_heads=2,
        digits_only=10,
    ),
    'standard_attn': dict(
        freeze_tok=False, freeze_pos=False, use_ffn=True,
        attn_type='standard', n_layers=1, n_heads=2,
        digits_only=None,
    ),
    'split_heads': dict(
        freeze_tok=False, freeze_pos=False, use_ffn=True,
        attn_type='split_heads', n_layers=1, n_heads=2,
        digits_only=None,
    ),
    '1head': dict(
        freeze_tok=False, freeze_pos=False, use_ffn=True,
        attn_type='qkpos_vtok', n_layers=1, n_heads=1,
        digits_only=None,
    ),
    'no_ffn_1head': dict(
        freeze_tok=False, freeze_pos=False, use_ffn=False,
        attn_type='qkpos_vtok', n_layers=1, n_heads=1,
        digits_only=None,
    ),
    'shared_qk': dict(
        freeze_tok=False, freeze_pos=False, use_ffn=True,
        attn_type='shared_qk', n_layers=1, n_heads=2,
        digits_only=None,
    ),
    'shared_qk_no_ffn': dict(
        freeze_tok=False, freeze_pos=False, use_ffn=False,
        attn_type='shared_qk', n_layers=1, n_heads=2,
        digits_only=None,
    ),
    'freeze_both_shared_qk_no_ffn': dict(
        freeze_tok=True, freeze_pos=True, use_ffn=False,
        attn_type='shared_qk', n_layers=1, n_heads=2,
        digits_only=None,
    ),
}


def run_experiment(exp_name, config, num_runs=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"Config: {config}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    digits_only = config.pop('digits_only', None)
    model_config = {k: v for k, v in config.items()}

    results = []
    for run in range(num_runs):
        if num_runs > 1:
            print(f"\n--- Run {run+1}/{num_runs} ---")

        model = ConfigurableTransformer(**model_config).to(device)
        n_trainable, n_total = model.count_parameters()
        print(f"Parameters: {n_trainable} trainable / {n_total} total (counted)")

        for name, p in model.named_parameters():
            print(f"  {name}: {p.shape} -> {p.numel()}")

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        # Fixed validation set
        val_dataset = AdditionDataset(VAL_SIZE, seed=42)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        best_exact_acc = -1.0
        best_epoch = 0
        patience_counter = 0
        epoch_at_99 = None
        epoch_at_995 = None
        history = []

        start_time = time.time()

        for epoch in range(1, EPOCHS + 1):
            train_dataset = AdditionDataset(TRAIN_SAMPLES_PER_EPOCH, seed=None, digits_only=digits_only)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

            model.train()
            epoch_loss = 0.0
            epoch_tokens = 0

            for tokens, loss_mask, _nd in train_loader:
                tokens = tokens.to(device)
                loss_mask = loss_mask.to(device)

                logits = model(tokens[:, :-1])
                targets = tokens[:, 1:]
                mask = loss_mask[:, 1:]

                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    reduction='none'
                ).reshape(targets.shape)

                masked_loss = (loss * mask).sum()
                n_tokens = mask.sum()

                if n_tokens > 0:
                    avg_loss = masked_loss / n_tokens
                    optimizer.zero_grad()
                    avg_loss.backward()
                    optimizer.step()
                    epoch_loss += masked_loss.item()
                    epoch_tokens += n_tokens.item()

            train_avg_loss = epoch_loss / max(epoch_tokens, 1)

            # Evaluate every 5 epochs for speed, every epoch near convergence
            if epoch <= 10 or epoch % 5 == 0 or patience_counter > PATIENCE - 20:
                val_loss, val_digit_acc, val_exact_acc, bucket_acc = evaluate(model, val_loader, device)
                bucket_str = " ".join(f"{nd}d:{acc:.3f}" for nd, acc in bucket_acc.items())

                elapsed = time.time() - start_time
                print(
                    f"Epoch {epoch:3d} [{elapsed:.0f}s] | "
                    f"train_loss={train_avg_loss:.4f} | "
                    f"val_loss={val_loss:.4f} | "
                    f"digit_acc={val_digit_acc:.4f} | "
                    f"exact={val_exact_acc:.4f} | "
                    f"{bucket_str}"
                )

                history.append({
                    'epoch': epoch,
                    'train_loss': train_avg_loss,
                    'val_loss': val_loss,
                    'digit_acc': val_digit_acc,
                    'exact_acc': val_exact_acc,
                    'bucket_acc': dict(bucket_acc),
                })

                if epoch_at_99 is None and val_exact_acc >= 0.99:
                    epoch_at_99 = epoch
                    print(f"  *** Reached 99% at epoch {epoch}! ***")
                if epoch_at_995 is None and val_exact_acc >= 0.995:
                    epoch_at_995 = epoch
                    print(f"  *** Reached 99.5% at epoch {epoch}! ***")

                if val_exact_acc > best_exact_acc:
                    best_exact_acc = val_exact_acc
                    best_epoch = epoch
                    patience_counter = 0
                    torch.save(model.state_dict(), f'exp_{exp_name}_best.pt')
                else:
                    patience_counter += 5 if epoch > 10 else 1

                if patience_counter >= PATIENCE:
                    print(f"Early stopping at epoch {epoch}.")
                    break
            else:
                patience_counter += 1

        total_time = time.time() - start_time

        # Final evaluation with larger test set
        print(f"\n--- Final Evaluation (run {run+1}) ---")
        model.load_state_dict(torch.load(f'exp_{exp_name}_best.pt', weights_only=True))
        final_val = AdditionDataset(10000, seed=999)
        final_loader = DataLoader(final_val, batch_size=BATCH_SIZE, shuffle=False)
        _, _, final_exact_acc, final_bucket = evaluate(model, final_loader, device)
        bucket_str = " ".join(f"{nd}d:{acc:.3f}" for nd, acc in final_bucket.items())

        result = {
            'run': run,
            'n_trainable': n_trainable,
            'n_total': n_total,
            'best_exact_acc': best_exact_acc,
            'best_epoch': best_epoch,
            'final_exact_acc': final_exact_acc,
            'final_bucket': final_bucket,
            'epoch_at_99': epoch_at_99,
            'epoch_at_995': epoch_at_995,
            'total_time': total_time,
        }
        results.append(result)

        print(f"Best: {best_exact_acc:.4f} at epoch {best_epoch}")
        print(f"Final (10K test): {final_exact_acc:.4f}")
        print(f"By length: {bucket_str}")
        print(f"Epoch@99%: {epoch_at_99}, Epoch@99.5%: {epoch_at_995}")
        print(f"Total time: {total_time:.1f}s")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {exp_name}")
    print(f"  Params: {results[0]['n_trainable']} trainable / {results[0]['n_total']} total")
    for r in results:
        print(f"  Run {r['run']}: final={r['final_exact_acc']:.4f}, "
              f"best={r['best_exact_acc']:.4f}@ep{r['best_epoch']}, "
              f"99%@ep{r['epoch_at_99']}, 99.5%@ep{r['epoch_at_995']}, "
              f"time={r['total_time']:.0f}s")
    print(f"{'='*60}\n")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, required=True,
                        choices=list(EXPERIMENTS.keys()) + ['all', 'freeze_ablation', 'ffn_ablation', 'data_ablation', 'attn_ablation'],
                        help='Experiment name or group')
    parser.add_argument('--runs', type=int, default=1, help='Number of runs per config')
    args = parser.parse_args()

    if args.exp == 'all':
        exp_names = list(EXPERIMENTS.keys())
    elif args.exp == 'freeze_ablation':
        exp_names = ['baseline', 'freeze_tok', 'freeze_pos', 'freeze_both']
    elif args.exp == 'ffn_ablation':
        exp_names = ['baseline', 'no_ffn', 'no_ffn_freeze_both']
    elif args.exp == 'data_ablation':
        exp_names = ['baseline', 'digits10_only']
    elif args.exp == 'attn_ablation':
        exp_names = ['baseline', 'standard_attn', 'split_heads', 'shared_qk']
    else:
        exp_names = [args.exp]

    all_results = {}
    for exp_name in exp_names:
        config = EXPERIMENTS[exp_name].copy()
        results = run_experiment(exp_name, config, num_runs=args.runs)
        all_results[exp_name] = results

    # Print comparison table
    print(f"\n{'='*80}")
    print("COMPARISON TABLE")
    print(f"{'='*80}")
    print(f"{'Experiment':<30} {'Params':>6} {'Final Acc':>10} {'Best Acc':>10} {'Ep@99%':>7} {'Ep@99.5%':>8}")
    print(f"{'-'*30} {'-'*6} {'-'*10} {'-'*10} {'-'*7} {'-'*8}")
    for exp_name, results in all_results.items():
        r = results[0]
        print(f"{exp_name:<30} {r['n_total']:>6} {r['final_exact_acc']:>10.4f} "
              f"{r['best_exact_acc']:>10.4f} {str(r['epoch_at_99']):>7} {str(r['epoch_at_995']):>8}")


if __name__ == '__main__':
    main()
