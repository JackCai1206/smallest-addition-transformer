"""Ablation training script for SLURM submission.

Usage:
    python train_ablation.py --variant baseline
    python train_ablation.py --variant no_ffn
    python train_ablation.py --variant d10
    python train_ablation.py --variant shqk
    python train_ablation.py --variant no_ffn_shqk
"""

import argparse, math, os, time, json
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict
from data import AdditionDataset
from config import (
    VOCAB_SIZE, D_MODEL, TOK_DIM, POS_DIM, N_HEADS, HEAD_DIM,
    FFN_DIM, MAX_SEQ_LEN, MAX_DIGITS, ANSWER_LEN, N_POS,
    X_START, PLUS_POS, Y_START, EQ_POS, Z_START, EOS_POS,
    PLUS_TOKEN, EQUALS_TOKEN, EOS_TOKEN, PAD_TOKEN,
    BATCH_SIZE, LR, EPOCHS, VAL_SIZE, TRAIN_SAMPLES_PER_EPOCH,
    LOG_EVERY, PATIENCE
)
from model import build_token_embedding, build_positional_encoding


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
    """Shared Q=K projection from positional dims, V from token dims."""
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

class AblationModel(nn.Module):
    def __init__(self, use_ffn=True, attn_type='qkpv', n_layers=1, n_heads=N_HEADS):
        super().__init__()
        # Always trainable embeddings (frozen doesn't work per prior experiments)
        self.tok_emb = nn.Embedding(VOCAB_SIZE, TOK_DIM)
        self.tok_emb.weight.data.copy_(build_token_embedding())
        self.pos_enc = nn.Parameter(build_positional_encoding())

        blocks = []
        for _ in range(n_layers):
            if attn_type == 'qkpv':
                attn = QKPosVTokAttention(n_heads)
            elif attn_type == 'shqk':
                attn = SharedQKAttention(n_heads)
            else:
                raise ValueError(f"Unknown attn_type: {attn_type}")
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
        tok = self.tok_emb(idx)
        pos = self.pos_enc[:T].unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([tok, pos], dim=-1)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)

    @torch.no_grad()
    def generate(self, prompt):
        self.eval()
        B, Tp = prompt.shape
        full_seq = torch.zeros(B, Tp + ANSWER_LEN + 1, dtype=torch.long, device=prompt.device)
        full_seq[:, :Tp] = prompt
        for step in range(ANSWER_LEN + 1):
            T = Tp + step
            logits = self.forward(full_seq[:, :T])
            full_seq[:, T] = logits[:, -1].argmax(dim=-1)
        return full_seq[:, Tp:]

    def count_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total


# ── Variants ──

VARIANTS = {
    'baseline':     dict(use_ffn=True,  attn_type='qkpv', n_layers=1, n_heads=2, digits_only=None),
    'no_ffn':       dict(use_ffn=False, attn_type='qkpv', n_layers=1, n_heads=2, digits_only=None),
    'no_ffn_2L':    dict(use_ffn=False, attn_type='qkpv', n_layers=2, n_heads=2, digits_only=None),
    'd10':          dict(use_ffn=True,  attn_type='qkpv', n_layers=1, n_heads=2, digits_only=10),
    'shqk':         dict(use_ffn=True,  attn_type='shqk', n_layers=1, n_heads=2, digits_only=None),
    'shqk_no_ffn':  dict(use_ffn=False, attn_type='shqk', n_layers=1, n_heads=2, digits_only=None),
    'no_ffn_1h':    dict(use_ffn=False, attn_type='qkpv', n_layers=1, n_heads=1, digits_only=None),
    '1h':           dict(use_ffn=True,  attn_type='qkpv', n_layers=1, n_heads=1, digits_only=None),
}


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = total_correct = total_tokens = total_exact = total_seq = 0
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
                targets.reshape(-1), reduction='none'
            ).reshape(targets.shape)

            masked_loss = (loss * mask).sum()
            n_tok = mask.sum()
            if n_tok > 0:
                total_loss += masked_loss.item()
                total_tokens += n_tok.item()

            preds = logits.argmax(dim=-1)
            total_correct += ((preds == targets) & (mask == 1)).sum().item()

            prompt = tokens[:, :EQ_POS + 1]
            answer_target = tokens[:, EQ_POS + 1:]
            generated = model.generate(prompt)

            exact = (generated == answer_target).all(dim=-1)
            total_exact += exact.sum().item()
            total_seq += tokens.size(0)

            for nd in range(1, MAX_DIGITS + 1):
                mask_nd = (n_digits == nd)
                if mask_nd.any():
                    bucket_total[nd] += mask_nd.sum().item()
                    bucket_correct[nd] += (exact & mask_nd.to(device)).sum().item()

    avg_loss = total_loss / max(total_tokens, 1)
    digit_acc = total_correct / max(total_tokens, 1)
    exact_acc = total_exact / max(total_seq, 1)
    bucket_acc = {nd: bucket_correct[nd] / max(bucket_total[nd], 1) for nd in sorted(bucket_total.keys())}
    return avg_loss, digit_acc, exact_acc, bucket_acc


def train(variant_name):
    cfg = VARIANTS[variant_name]
    digits_only = cfg.pop('digits_only', None)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Variant: {variant_name}")
    print(f"Config: {cfg}")
    print(f"Digits only: {digits_only}")
    print(f"Device: {device}")

    model = AblationModel(**cfg).to(device)
    n_trainable, n_total = model.count_parameters()
    print(f"Parameters: {n_trainable} trainable / {n_total} total")

    for name, p in model.named_parameters():
        print(f"  {name}: {list(p.shape)} = {p.numel()}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    val_dataset = AdditionDataset(VAL_SIZE, seed=42)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    best_exact_acc = -1.0
    best_epoch = 0
    patience_counter = 0
    save_name = f'best_ablation_{variant_name}.pt'

    for epoch in range(1, EPOCHS + 1):
        if digits_only:
            # Custom dataset for fixed digit count
            from quick_ablate import DS
            train_dataset = DS(TRAIN_SAMPLES_PER_EPOCH, seed=None, donly=digits_only)
        else:
            train_dataset = AdditionDataset(TRAIN_SAMPLES_PER_EPOCH, seed=None)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        model.train()
        epoch_loss = epoch_tokens = 0

        for tokens, loss_mask, _nd in train_loader:
            tokens = tokens.to(device)
            loss_mask = loss_mask.to(device)

            logits = model(tokens[:, :-1])
            targets = tokens[:, 1:]
            mask = loss_mask[:, 1:]

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1), reduction='none'
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

        val_loss, val_digit_acc, val_exact_acc, bucket_acc = evaluate(model, val_loader, device)
        bucket_str = " ".join(f"{nd}d:{acc:.3f}" for nd, acc in bucket_acc.items())

        print(
            f"Epoch {epoch:3d} | "
            f"train_loss={train_avg_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"digit_acc={val_digit_acc:.4f} | "
            f"exact_acc={val_exact_acc:.4f} | "
            f"by_len: {bucket_str}"
        )

        if val_exact_acc > best_exact_acc:
            best_exact_acc = val_exact_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), save_name)
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}. Best exact_acc={best_exact_acc:.4f} at epoch {best_epoch}")
            break

    print(f"\nTraining complete. Best val_exact_acc={best_exact_acc:.4f} at epoch {best_epoch}")

    # Final eval on bigger test set
    model.load_state_dict(torch.load(save_name, weights_only=True))
    final_ds = AdditionDataset(10000, seed=999)
    final_loader = DataLoader(final_ds, batch_size=BATCH_SIZE, shuffle=False)
    _, _, final_exact, final_bucket = evaluate(model, final_loader, device)
    bucket_str = " ".join(f"{nd}d:{acc:.3f}" for nd, acc in final_bucket.items())
    print(f"Final eval (10K): exact_acc={final_exact:.4f} | by_len: {bucket_str}")

    # Save results
    results = {
        'variant': variant_name,
        'params_trainable': n_trainable,
        'params_total': n_total,
        'best_exact_acc': best_exact_acc,
        'best_epoch': best_epoch,
        'final_exact_acc': final_exact,
        'config': {k: str(v) for k, v in VARIANTS.get(variant_name, cfg).items()},
    }
    with open(f'results_{variant_name}.json', 'w') as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, required=True, choices=list(VARIANTS.keys()))
    args = parser.parse_args()
    train(args.variant)


if __name__ == '__main__':
    main()
