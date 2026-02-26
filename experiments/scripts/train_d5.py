"""d_model=5 experiments: tok_dim=2 + pos_dim=3.

Key insight: reduce token embedding from 3D to 2D.
- 2D spiral (cos, sin) places digits 0-9 evenly on unit circle
- Circular structure is natural for modular arithmetic (addition mod 10)
- Saves params: tok_emb(14→28 vs 42), v_proj, head_proj, LNs, FFN

Parameter estimates:
  d5_f3: 286p  (tok_dim=2, pos_dim=3, ffn_dim=3)
  d5_f4: 297p  (tok_dim=2, pos_dim=3, ffn_dim=4)
  d5_f2: 275p  (tok_dim=2, pos_dim=3, ffn_dim=2)
  d5_f1: 264p  (tok_dim=2, pos_dim=3, ffn_dim=1)
  d4s_f3: 219p (tok_dim=2, pos_dim=2, ffn_dim=3)
  d4s_f4: 229p (tok_dim=2, pos_dim=2, ffn_dim=4)
"""

import argparse, math, os, json
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict
from data import AdditionDataset
from config import (
    VOCAB_SIZE, MAX_SEQ_LEN, MAX_DIGITS, ANSWER_LEN, EQ_POS,
    BATCH_SIZE, LR, VAL_SIZE, TRAIN_SAMPLES_PER_EPOCH,
    PLUS_TOKEN, EQUALS_TOKEN, EOS_TOKEN, PAD_TOKEN,
    X_START, PLUS_POS, Y_START, Z_START, EOS_POS, N_POS
)


def _spiral3(index, period):
    """3D spiral: (cos, sin, linear)."""
    return (
        math.cos(2 * math.pi * index / period),
        math.sin(2 * math.pi * index / period),
        index / max(period - 1, 1),
    )


def _spiral2(index, period):
    """2D spiral: (cos, sin) only."""
    return (
        math.cos(2 * math.pi * index / period),
        math.sin(2 * math.pi * index / period),
    )


def build_tok_emb_2d():
    """(VOCAB_SIZE, 2) — 2D circular embedding for tokens."""
    emb = torch.zeros(VOCAB_SIZE, 2)
    for d in range(10):
        emb[d, 0], emb[d, 1] = _spiral2(d, 10)
    emb[PLUS_TOKEN]   = torch.tensor([2.0, 0.0])
    emb[EQUALS_TOKEN] = torch.tensor([0.0, 2.0])
    emb[EOS_TOKEN]    = torch.tensor([-2.0, 0.0])
    emb[PAD_TOKEN]    = torch.tensor([0.0, -2.0])
    return emb


def build_pos_enc_3d():
    """(MAX_SEQ_LEN, 3) — standard 3D spiral positional encoding."""
    from model import build_positional_encoding
    return build_positional_encoding()


def build_pos_enc_2d():
    """(MAX_SEQ_LEN, 2) — 2D spiral positional encoding."""
    pe = torch.zeros(MAX_SEQ_LEN, 2)
    for i in range(MAX_DIGITS):
        pe[X_START + i, 0], pe[X_START + i, 1] = _spiral2(i, N_POS)
    for i in range(MAX_DIGITS):
        pe[Y_START + i, 0], pe[Y_START + i, 1] = _spiral2(i, N_POS)
    for i in range(min(ANSWER_LEN, N_POS)):
        pe[Z_START + i, 0], pe[Z_START + i, 1] = _spiral2(i, N_POS)
    pe[Z_START + 10] = torch.tensor([0.0, 1.5])
    pe[PLUS_POS] = torch.tensor([2.0, 0.0])
    pe[EQ_POS]   = torch.tensor([0.0, 2.0])
    pe[EOS_POS]  = torch.tensor([-2.0, 0.0])
    return pe


class D5Model(nn.Module):
    """Flexible tok_dim/pos_dim model with tied_proj output head."""
    def __init__(self, tok_dim=2, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=3):
        super().__init__()
        self.tok_dim = tok_dim
        self.pos_dim = pos_dim
        self.d_model = tok_dim + pos_dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.attn_dim = n_heads * head_dim

        # Embeddings
        self.tok_emb = nn.Embedding(VOCAB_SIZE, tok_dim)
        if tok_dim == 2:
            self.tok_emb.weight.data.copy_(build_tok_emb_2d())
        elif tok_dim == 3:
            from model import build_token_embedding
            self.tok_emb.weight.data.copy_(build_token_embedding())

        if pos_dim == 3:
            self.pos_enc = nn.Parameter(build_pos_enc_3d())
        elif pos_dim == 2:
            self.pos_enc = nn.Parameter(build_pos_enc_2d())

        # Attention (QK from pos, V from tok)
        self.q_proj = nn.Linear(pos_dim, self.attn_dim, bias=False)
        self.k_proj = nn.Linear(pos_dim, self.attn_dim, bias=False)
        self.v_proj = nn.Linear(tok_dim, self.attn_dim, bias=False)
        self.out_proj = nn.Linear(self.attn_dim, self.d_model, bias=False)

        # Norms (WITH biases — essential!)
        self.ln1 = nn.LayerNorm(self.d_model)
        self.ln2 = nn.LayerNorm(self.d_model)
        self.ln_f = nn.LayerNorm(self.d_model)

        # FFN (with biases)
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, self.d_model),
        )

        # Tied output head: Linear(d_model → tok_dim) then @ tok_emb.T
        self.head_proj = nn.Linear(self.d_model, tok_dim, bias=False)

        self.register_buffer("causal_mask",
            torch.tril(torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN)).unsqueeze(0).unsqueeze(0))

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

        # Pre-norm attention
        h = self.ln1(x)
        x_pos = h[:, :, self.tok_dim:]
        x_tok = h[:, :, :self.tok_dim]
        q = self.q_proj(x_pos).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_pos).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_tok).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, self.attn_dim)
        x = x + self.out_proj(out)

        # FFN
        x = x + self.ffn(self.ln2(x))
        x = self.ln_f(x)

        # Tied output
        return self.head_proj(x) @ self.tok_emb.weight.T

    @torch.no_grad()
    def generate(self, prompt):
        self.eval()
        B, Tp = prompt.shape
        fs = torch.zeros(B, Tp + ANSWER_LEN + 1, dtype=torch.long, device=prompt.device)
        fs[:, :Tp] = prompt
        for s in range(ANSWER_LEN + 1):
            T = Tp + s
            lg = self.forward(fs[:, :T])
            fs[:, T] = lg[:, -1].argmax(-1)
        return fs[:, Tp:]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad), \
               sum(p.numel() for p in self.parameters())


VARIANTS = {
    # === d_model=5 (tok_dim=2, pos_dim=3) ===
    'd5_f4': dict(tok_dim=2, pos_dim=3, ffn_dim=4, n_heads=2, head_dim=3),
    'd5_f3': dict(tok_dim=2, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=3),
    'd5_f2': dict(tok_dim=2, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3),
    'd5_f1': dict(tok_dim=2, pos_dim=3, ffn_dim=1, n_heads=2, head_dim=3),

    # === d_model=5 with head_dim=2 ===
    'd5_f3_hd2': dict(tok_dim=2, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=2),
    'd5_f4_hd2': dict(tok_dim=2, pos_dim=3, ffn_dim=4, n_heads=2, head_dim=2),

    # === d_model=4 split-dim (tok=2, pos=2) ===
    'd4s_f4': dict(tok_dim=2, pos_dim=2, ffn_dim=4, n_heads=2, head_dim=3),
    'd4s_f3': dict(tok_dim=2, pos_dim=2, ffn_dim=3, n_heads=2, head_dim=3),
    'd4s_f3_hd2': dict(tok_dim=2, pos_dim=2, ffn_dim=3, n_heads=2, head_dim=2),

    # === Reference: tp_f3 equivalent at d=6 ===
    'd6_f3_ref': dict(tok_dim=3, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=3),
}


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = total_correct = total_tokens = total_exact = total_seq = 0
    bucket_correct = defaultdict(int)
    bucket_total = defaultdict(int)
    with torch.no_grad():
        for tokens, loss_mask, n_digits in dataloader:
            tokens, loss_mask = tokens.to(device), loss_mask.to(device)
            logits = model(tokens[:, :-1])
            targets = tokens[:, 1:]
            mask = loss_mask[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                                   reduction='none').reshape(targets.shape)
            masked_loss = (loss * mask).sum()
            n_tok = mask.sum()
            if n_tok > 0:
                total_loss += masked_loss.item()
                total_tokens += n_tok.item()
            total_correct += ((logits.argmax(-1) == targets) & (mask == 1)).sum().item()
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
    return (total_loss / max(total_tokens, 1),
            total_correct / max(total_tokens, 1),
            total_exact / max(total_seq, 1),
            {nd: bucket_correct[nd] / max(bucket_total[nd], 1) for nd in sorted(bucket_total.keys())})


def train(variant_name):
    cfg = VARIANTS[variant_name]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Variant: {variant_name}")
    print(f"Config: {cfg}")
    print(f"Device: {device}")

    model = D5Model(**cfg).to(device)
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
    patience = 500
    save_name = f'best_d5_{variant_name}.pt'

    for epoch in range(1, 1001):
        train_dataset = AdditionDataset(TRAIN_SAMPLES_PER_EPOCH, seed=None)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        model.train()
        epoch_loss = epoch_tokens = 0
        for tokens, loss_mask, _nd in train_loader:
            tokens, loss_mask = tokens.to(device), loss_mask.to(device)
            logits = model(tokens[:, :-1])
            targets = tokens[:, 1:]
            mask = loss_mask[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                                   reduction='none').reshape(targets.shape)
            masked_loss = (loss * mask).sum()
            n_tokens = mask.sum()
            if n_tokens > 0:
                (masked_loss / n_tokens).backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += masked_loss.item()
                epoch_tokens += n_tokens.item()

        val_loss, val_da, val_ea, bucket = evaluate(model, val_loader, device)
        bucket_str = " ".join(f"{nd}d:{acc:.3f}" for nd, acc in bucket.items())
        print(f"Epoch {epoch:3d} | train_loss={epoch_loss/max(epoch_tokens,1):.4f} | "
              f"val_loss={val_loss:.4f} | da={val_da:.4f} | ea={val_ea:.4f} | {bucket_str}")

        if val_ea > best_exact_acc:
            best_exact_acc = val_ea
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), save_name)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stop at epoch {epoch}. Best ea={best_exact_acc:.4f} at epoch {best_epoch}")
            break

    print(f"Training done. Best ea={best_exact_acc:.4f} at epoch {best_epoch}")

    if os.path.exists(save_name):
        model.load_state_dict(torch.load(save_name, weights_only=True))
    for seed_name, seed in [("seed=999", 999), ("seed=2025", 2025)]:
        _, _, final_ea, final_bucket = evaluate(model,
            DataLoader(AdditionDataset(10000, seed=seed), batch_size=BATCH_SIZE), device)
        bucket_str = " ".join(f"{nd}d:{acc:.3f}" for nd, acc in final_bucket.items())
        print(f"Final (10K, {seed_name}): ea={final_ea:.4f} | {bucket_str}")

    with open(f'results_d5_{variant_name}.json', 'w') as f:
        json.dump({'variant': variant_name, 'params': n_total, 'best_ea': best_exact_acc,
                   'best_epoch': best_epoch, 'config': cfg}, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, required=True, choices=list(VARIANTS.keys()))
    args = parser.parse_args()
    train(args.variant)


if __name__ == '__main__':
    main()
