"""Incremental parameter reduction experiments.

Strategy: Instead of removing entire components, make each component SMALLER.
The baseline (438 params) works. What's the smallest model that still converges?

Key savings targets:
1. Remove biases from FFN and LN (save 30 params)
2. Smaller FFN dim (4 or 3 instead of 6)
3. Factored output head (rank-2 or rank-3)
4. Factored position encoding (digit + segment decomposition)
"""

import argparse, math, os, json
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict
from data import AdditionDataset
from config import (
    VOCAB_SIZE, D_MODEL, TOK_DIM, POS_DIM, N_HEADS, HEAD_DIM,
    FFN_DIM, MAX_SEQ_LEN, MAX_DIGITS, ANSWER_LEN,
    EQ_POS, BATCH_SIZE, LR, EPOCHS, VAL_SIZE,
    TRAIN_SAMPLES_PER_EPOCH, PATIENCE
)
from model import build_token_embedding, build_positional_encoding


class BiasFreeLN(nn.Module):
    """LayerNorm with mean centering but no bias parameter. Saves 6 params per LN."""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return self.weight * (x - mean) / torch.sqrt(var + self.eps)


class CausalSelfAttention(nn.Module):
    """QK from pos dims, V from tok dims, with out_proj."""
    def __init__(self):
        super().__init__()
        self.n_heads = N_HEADS
        self.head_dim = HEAD_DIM
        self.q_proj = nn.Linear(POS_DIM, N_HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(POS_DIM, N_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(TOK_DIM, N_HEADS * HEAD_DIM, bias=False)
        self.out_proj = nn.Linear(N_HEADS * HEAD_DIM, D_MODEL, bias=False)
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


class ShrinkModel(nn.Module):
    def __init__(self, no_bias=False, ffn_dim=6, head_rank=0, factored_pos=False):
        """
        Args:
            no_bias: If True, remove biases from FFN and use BiasFreeLN
            ffn_dim: FFN intermediate dimension (6=baseline, 4, 3, etc.)
            head_rank: If >0, use factored output head with this rank. 0=normal.
            factored_pos: If True, use digit+segment factored position encoding
        """
        super().__init__()
        self.head_rank = head_rank
        self.factored_pos = factored_pos

        # Token embedding (always trainable, spiral-init)
        self.tok_emb = nn.Embedding(VOCAB_SIZE, TOK_DIM)
        self.tok_emb.weight.data.copy_(build_token_embedding())

        # Position encoding
        if factored_pos:
            # Factored: digit_pos(11,3) + segment(3,3) + special(3,3) = 51 params
            self.digit_enc = nn.Parameter(torch.zeros(11, POS_DIM))
            self.segment_enc = nn.Parameter(torch.zeros(3, POS_DIM))
            self.special_enc = nn.Parameter(torch.zeros(3, POS_DIM))
            self._init_factored_pos()
            # Precompute position mapping (buffer, not params)
            digit_idx, segment_idx, is_special, special_idx = [], [], [], []
            for p in range(MAX_SEQ_LEN):
                if p < 10:       # X digits 0-9
                    digit_idx.append(p); segment_idx.append(0); is_special.append(False); special_idx.append(0)
                elif p == 10:    # +
                    digit_idx.append(0); segment_idx.append(0); is_special.append(True); special_idx.append(0)
                elif p < 21:     # Y digits 0-9
                    digit_idx.append(p - 11); segment_idx.append(1); is_special.append(False); special_idx.append(0)
                elif p == 21:    # =
                    digit_idx.append(0); segment_idx.append(0); is_special.append(True); special_idx.append(1)
                elif p < 33:     # Z digits 0-10
                    digit_idx.append(p - 22); segment_idx.append(2); is_special.append(False); special_idx.append(0)
                else:            # EOS
                    digit_idx.append(0); segment_idx.append(0); is_special.append(True); special_idx.append(2)
            self.register_buffer('_digit_idx', torch.tensor(digit_idx))
            self.register_buffer('_segment_idx', torch.tensor(segment_idx))
            self.register_buffer('_is_special', torch.tensor(is_special))
            self.register_buffer('_special_idx', torch.tensor(special_idx))
        else:
            self.pos_enc = nn.Parameter(build_positional_encoding())

        # Attention
        self.attn = CausalSelfAttention()

        # Norms
        NormClass = BiasFreeLN if no_bias else nn.LayerNorm
        self.ln1 = NormClass(D_MODEL)
        self.ln2 = NormClass(D_MODEL)
        self.ln_f = NormClass(D_MODEL)

        # FFN
        if no_bias:
            self.ffn = nn.Sequential(
                nn.Linear(D_MODEL, ffn_dim, bias=False),
                nn.GELU(),
                nn.Linear(ffn_dim, D_MODEL, bias=False),
            )
        else:
            self.ffn = nn.Sequential(
                nn.Linear(D_MODEL, ffn_dim),
                nn.GELU(),
                nn.Linear(ffn_dim, D_MODEL),
            )

        # Output head
        if head_rank > 0:
            self.head_w1 = nn.Linear(D_MODEL, head_rank, bias=False)
            self.head_w2 = nn.Linear(head_rank, VOCAB_SIZE, bias=False)
        else:
            self.head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)

        self._init_weights()

    def _init_factored_pos(self):
        """Initialize factored pos encoding from spiral values."""
        full_pos = build_positional_encoding()  # (34, 3)
        # Initialize digit_enc as the average digit encoding
        for d in range(11):
            positions = []
            if d < 10:
                positions.extend([d, 11 + d])  # X and Y digit d
            if d <= 10:
                positions.append(22 + d)  # Z digit d
            if positions:
                self.digit_enc.data[d] = full_pos[positions].mean(0)
        # Initialize segment_enc as the residual
        for seg, offset in enumerate([0, 11, 22]):
            n_digits = 10 if seg < 2 else 11
            residuals = []
            for d in range(n_digits):
                residuals.append(full_pos[offset + d] - self.digit_enc.data[d])
            self.segment_enc.data[seg] = torch.stack(residuals).mean(0)
        # Special tokens
        self.special_enc.data[0] = full_pos[10]   # +
        self.special_enc.data[1] = full_pos[21]   # =
        self.special_enc.data[2] = full_pos[33]   # EOS

    def _init_weights(self):
        for name, p in self.named_parameters():
            if name in ('tok_emb.weight', 'pos_enc', 'digit_enc', 'segment_enc', 'special_enc'):
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _get_pos_enc(self, T):
        if self.factored_pos:
            pos = self.digit_enc[self._digit_idx[:T]] + self.segment_enc[self._segment_idx[:T]]
            # Override special positions
            special_mask = self._is_special[:T]
            if special_mask.any():
                special_positions = special_mask.nonzero(as_tuple=True)[0]
                for sp in special_positions:
                    pos[sp] = self.special_enc[self._special_idx[sp]]
            return pos
        else:
            return self.pos_enc[:T]

    def forward(self, idx):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self._get_pos_enc(T).unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([tok, pos], dim=-1)

        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        x = self.ln_f(x)

        if self.head_rank > 0:
            return self.head_w2(self.head_w1(x))
        else:
            return self.head(x)

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


# Experiment configurations
VARIANTS = {
    # 1. No biases (save 30 params): 408p
    'nb': dict(no_bias=True, ffn_dim=6, head_rank=0, factored_pos=False),

    # 2. FFN dim=4 (save 26 params): 412p
    'ffn4': dict(no_bias=False, ffn_dim=4, head_rank=0, factored_pos=False),

    # 3. FFN dim=3 (save 39 params): 399p
    'ffn3': dict(no_bias=False, ffn_dim=3, head_rank=0, factored_pos=False),

    # 4. No bias + FFN4 (save 54 params): 384p
    'nb_ffn4': dict(no_bias=True, ffn_dim=4, head_rank=0, factored_pos=False),

    # 5. No bias + FFN3 (save 66 params): 372p
    'nb_ffn3': dict(no_bias=True, ffn_dim=3, head_rank=0, factored_pos=False),

    # 6. Factored head rank 2 (save 44 params): 394p
    'fh2': dict(no_bias=False, ffn_dim=6, head_rank=2, factored_pos=False),

    # 7. Factored head rank 3 (save 24 params): 414p
    'fh3': dict(no_bias=False, ffn_dim=6, head_rank=3, factored_pos=False),

    # 8. Factored position encoding (save 51 params): 387p
    'fpos': dict(no_bias=False, ffn_dim=6, head_rank=0, factored_pos=True),

    # 9. Combo: no bias + FFN4 + factored head r2 (save 98 params): 340p
    'nb_ffn4_fh2': dict(no_bias=True, ffn_dim=4, head_rank=2, factored_pos=False),

    # 10. Combo: no bias + FFN3 + factored head r2 (save 110 params): 328p
    'nb_ffn3_fh2': dict(no_bias=True, ffn_dim=3, head_rank=2, factored_pos=False),

    # 11. Combo: no bias + FFN4 + factored pos (save 105 params): 333p
    'nb_ffn4_fpos': dict(no_bias=True, ffn_dim=4, head_rank=0, factored_pos=True),

    # 12. All optimizations: no bias + FFN3 + factored head r2 + factored pos: 277p
    'all_shrink': dict(no_bias=True, ffn_dim=3, head_rank=2, factored_pos=True),
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

    model = ShrinkModel(**cfg).to(device)
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
    save_name = f'best_shrink_{variant_name}.pt'

    for epoch in range(1, EPOCHS + 1):
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

        if patience_counter >= PATIENCE:
            print(f"Early stop at epoch {epoch}. Best ea={best_exact_acc:.4f} at epoch {best_epoch}")
            break

    print(f"Training done. Best ea={best_exact_acc:.4f} at epoch {best_epoch}")

    model.load_state_dict(torch.load(save_name, weights_only=True))
    _, _, final_ea, final_bucket = evaluate(model, DataLoader(AdditionDataset(10000, seed=999), batch_size=BATCH_SIZE), device)
    bucket_str = " ".join(f"{nd}d:{acc:.3f}" for nd, acc in final_bucket.items())
    print(f"Final (10K): ea={final_ea:.4f} | {bucket_str}")

    with open(f'results_shrink_{variant_name}.json', 'w') as f:
        json.dump({'variant': variant_name, 'params': n_total, 'best_ea': best_exact_acc,
                   'best_epoch': best_epoch, 'final_ea': final_ea}, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, required=True, choices=list(VARIANTS.keys()))
    args = parser.parse_args()
    train(args.variant)


if __name__ == '__main__':
    main()
