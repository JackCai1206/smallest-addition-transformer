"""Push to beat the 311p leaderboard record.

Building on confirmed winners:
- tied_proj output head (saves 66p)
- FFN_DIM=3 works (saves 39p)
- Normal LN (with biases) is essential

Now testing:
- FFN_DIM=2 (saves 52p from baseline)
- HEAD_DIM=2 instead of 3 (saves 30p in attention)
- Combinations targeting < 311 params
"""

import argparse, math, os, json
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict
from data import AdditionDataset
from config import (
    VOCAB_SIZE, D_MODEL, TOK_DIM, POS_DIM, MAX_SEQ_LEN,
    MAX_DIGITS, ANSWER_LEN, EQ_POS, BATCH_SIZE, LR, VAL_SIZE,
    TRAIN_SAMPLES_PER_EPOCH
)
from model import build_token_embedding, build_positional_encoding


class PushModel(nn.Module):
    """Configurable HEAD_DIM and FFN_DIM with tied_proj output."""
    def __init__(self, ffn_dim=3, head_dim=3, n_heads=2):
        super().__init__()
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.attn_dim = n_heads * head_dim  # total attention output dimension

        # Embeddings (always trainable, spiral-init)
        self.tok_emb = nn.Embedding(VOCAB_SIZE, TOK_DIM)
        self.tok_emb.weight.data.copy_(build_token_embedding())
        self.pos_enc = nn.Parameter(build_positional_encoding())

        # Attention (QK from pos, V from tok)
        self.q_proj = nn.Linear(POS_DIM, self.attn_dim, bias=False)
        self.k_proj = nn.Linear(POS_DIM, self.attn_dim, bias=False)
        self.v_proj = nn.Linear(TOK_DIM, self.attn_dim, bias=False)
        self.out_proj = nn.Linear(self.attn_dim, D_MODEL, bias=False)

        # Norms (WITH biases — essential!)
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.ln_f = nn.LayerNorm(D_MODEL)

        # FFN (with biases)
        self.ffn = nn.Sequential(
            nn.Linear(D_MODEL, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, D_MODEL),
        )

        # Tied output head
        self.head_proj = nn.Linear(D_MODEL, TOK_DIM, bias=False)

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
        x_pos = h[:, :, TOK_DIM:]
        x_tok = h[:, :, :TOK_DIM]
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
    # Reference: tp_f3 (what's currently converging)
    'tp_f3_ref': dict(ffn_dim=3, head_dim=3, n_heads=2),  # 333p

    # FFN_DIM=2 (save 13 more from tp_f3)
    'tp_f2': dict(ffn_dim=2, head_dim=3, n_heads=2),  # 320p

    # HEAD_DIM=2 with FFN3
    'tp_f3_hd2': dict(ffn_dim=3, head_dim=2, n_heads=2),  # 303p

    # HEAD_DIM=2 with FFN2 (most aggressive)
    'tp_f2_hd2': dict(ffn_dim=2, head_dim=2, n_heads=2),  # 290p

    # HEAD_DIM=2 with FFN4
    'tp_f4_hd2': dict(ffn_dim=4, head_dim=2, n_heads=2),  # 316p

    # FFN_DIM=1 (extreme)
    'tp_f1': dict(ffn_dim=1, head_dim=3, n_heads=2),  # 307p!

    # FFN_DIM=1 + HEAD_DIM=2
    'tp_f1_hd2': dict(ffn_dim=1, head_dim=2, n_heads=2),  # 277p!!
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

    model = PushModel(**cfg).to(device)
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
    save_name = f'best_push_{variant_name}.pt'

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

    with open(f'results_push_{variant_name}.json', 'w') as f:
        json.dump({'variant': variant_name, 'params': n_total, 'best_ea': best_exact_acc,
                   'best_epoch': best_epoch}, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, required=True, choices=list(VARIANTS.keys()))
    args = parser.parse_args()
    train(args.variant)


if __name__ == '__main__':
    main()
