"""Training with tied/shared output head variants.

Tests aggressive parameter reduction by reusing tok_emb for the output head.
"""

import argparse, math, os, time, json
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
        x_pos, x_tok = x[:, :, TOK_DIM:], x[:, :, :TOK_DIM]
        q = self.q_proj(x_pos).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_pos).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_tok).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        a = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        a = a.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        a = F.softmax(a, dim=-1)
        out = (a @ v).transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
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
        x_pos, x_tok = x[:, :, TOK_DIM:], x[:, :, :TOK_DIM]
        qk = self.qk_proj(x_pos).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_tok).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        a = (qk @ qk.transpose(-2, -1)) / math.sqrt(self.head_dim)
        a = a.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        a = F.softmax(a, dim=-1)
        out = (a @ v).transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.out_proj(out)


class BlockWithFFN(nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = attn
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.ffn = nn.Sequential(nn.Linear(D_MODEL, FFN_DIM), nn.GELU(), nn.Linear(FFN_DIM, D_MODEL))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class BlockNoFFN(nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.ln = nn.LayerNorm(D_MODEL)
        self.attn = attn

    def forward(self, x):
        return x + self.attn(self.ln(x))


class TiedHeadModel(nn.Module):
    """Model with output head tied to token embeddings.

    Output projection: x -> proj(x) -> matmul with tok_emb.weight.T
    This saves params by sharing tok_emb between input and output.

    head_type:
      'tied_proj': logits = Linear(6->3)(x) @ tok_emb.T  (18 params for proj)
      'tied_tok':  logits = x[:,:,:3] @ tok_emb.T         (0 extra params)
      'normal':    logits = Linear(6->14)(x)               (84 params, baseline)
    """

    def __init__(self, head_type='tied_proj', use_ffn=True, attn_type='qkpv',
                 n_layers=1, n_heads=N_HEADS, use_final_ln=True):
        super().__init__()
        self.head_type = head_type
        self.use_final_ln = use_final_ln

        # Always trainable embeddings
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
                raise ValueError(attn_type)
            blocks.append(BlockWithFFN(attn) if use_ffn else BlockNoFFN(attn))
        self.blocks = nn.ModuleList(blocks)

        if use_final_ln:
            self.ln_f = nn.LayerNorm(D_MODEL)

        # Output head
        if head_type == 'tied_proj':
            self.head_proj = nn.Linear(D_MODEL, TOK_DIM, bias=False)  # 18 params
        elif head_type == 'tied_tok':
            pass  # No extra params — use tok dims directly
        elif head_type == 'normal':
            self.head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)  # 84 params
        else:
            raise ValueError(head_type)

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
        if self.use_final_ln:
            x = self.ln_f(x)

        # Compute logits
        if self.head_type == 'tied_proj':
            h = self.head_proj(x)  # (B, T, 3)
            logits = h @ self.tok_emb.weight.T  # (B, T, 14)
        elif self.head_type == 'tied_tok':
            h = x[:, :, :TOK_DIM]  # (B, T, 3) — just take token dims
            logits = h @ self.tok_emb.weight.T  # (B, T, 14)
        else:
            logits = self.head(x)
        return logits

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


VARIANTS = {
    # Tied head with projection (saves 66 params)
    'tied_proj':           dict(head_type='tied_proj', use_ffn=True,  attn_type='qkpv', n_layers=1, n_heads=2, use_final_ln=True),
    'tied_proj_no_ffn':    dict(head_type='tied_proj', use_ffn=False, attn_type='qkpv', n_layers=1, n_heads=2, use_final_ln=True),
    'tied_proj_shqk':      dict(head_type='tied_proj', use_ffn=True,  attn_type='shqk', n_layers=1, n_heads=2, use_final_ln=True),
    'tied_proj_shqk_nf':   dict(head_type='tied_proj', use_ffn=False, attn_type='shqk', n_layers=1, n_heads=2, use_final_ln=True),

    # Tied head using tok dims directly (saves 84 params!)
    'tied_tok':            dict(head_type='tied_tok',  use_ffn=True,  attn_type='qkpv', n_layers=1, n_heads=2, use_final_ln=True),
    'tied_tok_no_ffn':     dict(head_type='tied_tok',  use_ffn=False, attn_type='qkpv', n_layers=1, n_heads=2, use_final_ln=True),
    'tied_tok_shqk_nf':    dict(head_type='tied_tok',  use_ffn=False, attn_type='shqk', n_layers=1, n_heads=2, use_final_ln=True),

    # No final LN (saves another 12 params)
    'tied_proj_no_ln':     dict(head_type='tied_proj', use_ffn=True,  attn_type='qkpv', n_layers=1, n_heads=2, use_final_ln=False),
    'tied_tok_no_ffn_no_ln': dict(head_type='tied_tok', use_ffn=False, attn_type='qkpv', n_layers=1, n_heads=2, use_final_ln=False),
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

    model = TiedHeadModel(**cfg).to(device)
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
    save_name = f'best_tied_{variant_name}.pt'

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

    with open(f'results_tied_{variant_name}.json', 'w') as f:
        json.dump({'variant': variant_name, 'params': n_total, 'best_ea': best_exact_acc,
                   'best_epoch': best_epoch, 'final_ea': final_ea}, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, required=True, choices=list(VARIANTS.keys()))
    args = parser.parse_args()
    train(args.variant)


if __name__ == '__main__':
    main()
