"""Build on the tied_proj success (372p, 99.78% on AdderBoard).

Now combine tied_proj with additional parameter savings:
1. No biases (LN → BiasFreeLN, FFN no bias)
2. Smaller FFN (dim=4, dim=3)
3. Factored position encoding
4. Curriculum learning

The goal: minimize params while preserving the grokking convergence.
"""

import argparse, math, os, json
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict
import numpy as np
from data import AdditionDataset
from config import (
    VOCAB_SIZE, D_MODEL, TOK_DIM, POS_DIM, N_HEADS, HEAD_DIM,
    MAX_SEQ_LEN, MAX_DIGITS, ANSWER_LEN,
    EQ_POS, BATCH_SIZE, LR, VAL_SIZE,
    PLUS_TOKEN, EQUALS_TOKEN, EOS_TOKEN, PAD_TOKEN,
    TRAIN_SAMPLES_PER_EPOCH
)
from model import build_token_embedding, build_positional_encoding


class BiasFreeLN(nn.Module):
    """LayerNorm with mean centering but no bias parameter."""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return self.weight * (x - mean) / torch.sqrt(var + self.eps)


class WinnerModel(nn.Module):
    """tied_proj architecture with configurable savings."""
    def __init__(self, no_bias=False, ffn_dim=6, factored_pos=False,
                 norm_type='ln'):
        super().__init__()
        self.factored_pos = factored_pos

        # Token embedding (used for both input and output via tied_proj)
        self.tok_emb = nn.Embedding(VOCAB_SIZE, TOK_DIM)
        self.tok_emb.weight.data.copy_(build_token_embedding())

        # Position encoding
        if factored_pos:
            # Digit(11,3) + segment(3,3) + special(3,3) = 51 params (vs 102)
            self.digit_enc = nn.Parameter(torch.zeros(11, POS_DIM))
            self.segment_enc = nn.Parameter(torch.zeros(3, POS_DIM))
            self.special_enc = nn.Parameter(torch.zeros(3, POS_DIM))
            self._init_factored_pos()
            digit_idx, segment_idx, is_special, special_idx = [], [], [], []
            for p in range(MAX_SEQ_LEN):
                if p < 10:
                    digit_idx.append(p); segment_idx.append(0)
                    is_special.append(False); special_idx.append(0)
                elif p == 10:
                    digit_idx.append(0); segment_idx.append(0)
                    is_special.append(True); special_idx.append(0)
                elif p < 21:
                    digit_idx.append(p-11); segment_idx.append(1)
                    is_special.append(False); special_idx.append(0)
                elif p == 21:
                    digit_idx.append(0); segment_idx.append(0)
                    is_special.append(True); special_idx.append(1)
                elif p < 33:
                    digit_idx.append(p-22); segment_idx.append(2)
                    is_special.append(False); special_idx.append(0)
                else:
                    digit_idx.append(0); segment_idx.append(0)
                    is_special.append(True); special_idx.append(2)
            self.register_buffer('_digit_idx', torch.tensor(digit_idx))
            self.register_buffer('_segment_idx', torch.tensor(segment_idx))
            self.register_buffer('_is_special', torch.tensor(is_special))
            self.register_buffer('_special_idx', torch.tensor(special_idx))
        else:
            self.pos_enc = nn.Parameter(build_positional_encoding())

        # Attention (QK from pos, V from tok)
        self.q_proj = nn.Linear(POS_DIM, N_HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(POS_DIM, N_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(TOK_DIM, N_HEADS * HEAD_DIM, bias=False)
        self.out_proj = nn.Linear(N_HEADS * HEAD_DIM, D_MODEL, bias=False)

        # Norms
        NormClass = BiasFreeLN if no_bias or norm_type == 'bfln' else nn.LayerNorm
        self.ln1 = NormClass(D_MODEL)
        self.ln2 = NormClass(D_MODEL)
        self.ln_f = NormClass(D_MODEL)

        # FFN
        use_bias = not no_bias
        self.ffn = nn.Sequential(
            nn.Linear(D_MODEL, ffn_dim, bias=use_bias),
            nn.GELU(),
            nn.Linear(ffn_dim, D_MODEL, bias=use_bias),
        )

        # Tied output head: Linear(6→3) then matmul with tok_emb.T
        self.head_proj = nn.Linear(D_MODEL, TOK_DIM, bias=False)

        self.register_buffer("causal_mask",
            torch.tril(torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN)).unsqueeze(0).unsqueeze(0))

        self._init_weights()

    def _init_factored_pos(self):
        full_pos = build_positional_encoding()
        for d in range(11):
            positions = []
            if d < 10: positions.extend([d, 11+d])
            if d <= 10: positions.append(22+d)
            if positions:
                self.digit_enc.data[d] = full_pos[positions].mean(0)
        for seg, offset in enumerate([0, 11, 22]):
            n_digits = 10 if seg < 2 else 11
            residuals = []
            for d in range(n_digits):
                residuals.append(full_pos[offset+d] - self.digit_enc.data[d])
            self.segment_enc.data[seg] = torch.stack(residuals).mean(0)
        self.special_enc.data[0] = full_pos[10]   # +
        self.special_enc.data[1] = full_pos[21]   # =
        self.special_enc.data[2] = full_pos[33]   # EOS

    def _init_weights(self):
        for name, p in self.named_parameters():
            if name in ('tok_emb.weight', 'pos_enc', 'digit_enc', 'segment_enc', 'special_enc'):
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _get_pos(self, T):
        if self.factored_pos:
            pos = self.digit_enc[self._digit_idx[:T]] + self.segment_enc[self._segment_idx[:T]]
            special_mask = self._is_special[:T]
            if special_mask.any():
                for sp in special_mask.nonzero(as_tuple=True)[0]:
                    pos[sp] = self.special_enc[self._special_idx[sp]]
            return pos
        return self.pos_enc[:T]

    def forward(self, idx):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self._get_pos(T).unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([tok, pos], dim=-1)

        # Pre-norm attention
        h = self.ln1(x)
        x_pos = h[:, :, TOK_DIM:]
        x_tok = h[:, :, :TOK_DIM]
        q = self.q_proj(x_pos).view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2)
        k = self.k_proj(x_pos).view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.v_proj(x_tok).view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(HEAD_DIM)
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, N_HEADS * HEAD_DIM)
        x = x + self.out_proj(out)

        # Pre-norm FFN
        x = x + self.ffn(self.ln2(x))
        x = self.ln_f(x)

        # Tied output: project to tok_dim, then matmul with tok_emb
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


# Curriculum data generation
def generate_curriculum_batch(num_samples, max_digits, rng):
    n = rng.integers(1, max_digits + 1, size=num_samples)
    lo = np.where(n == 1, 0, 10 ** (n - 1))
    hi = 10 ** n
    x = (rng.random(num_samples) * (hi - lo) + lo).astype(np.int64)
    y = (rng.random(num_samples) * (hi - lo) + lo).astype(np.int64)
    z = x + y
    tokens = np.empty((num_samples, MAX_SEQ_LEN), dtype=np.int64)
    tmp = x.copy()
    for d in range(MAX_DIGITS):
        tokens[:, d] = tmp % 10; tmp //= 10
    tokens[:, MAX_DIGITS] = 10  # PLUS
    tmp = y.copy()
    for d in range(MAX_DIGITS):
        tokens[:, MAX_DIGITS+1+d] = tmp % 10; tmp //= 10
    tokens[:, 2*MAX_DIGITS+1] = 11  # EQUALS
    tmp = z.copy()
    for d in range(MAX_DIGITS+1):
        tokens[:, 2*MAX_DIGITS+2+d] = tmp % 10; tmp //= 10
    tokens[:, -1] = 12  # EOS
    loss_mask = np.array([0]*22 + [1]*12, dtype=np.int64)
    return tokens, np.tile(loss_mask, (num_samples, 1)), n


class CurriculumDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples, max_digits, seed=None):
        rng = np.random.default_rng(seed)
        tokens, loss_mask, n_digits = generate_curriculum_batch(num_samples, max_digits, rng)
        self.tokens = torch.from_numpy(tokens)
        self.loss_mask = torch.from_numpy(loss_mask)
        self.n_digits = torch.from_numpy(n_digits)
    def __len__(self): return len(self.tokens)
    def __getitem__(self, idx): return self.tokens[idx], self.loss_mask[idx], self.n_digits[idx]


def get_curriculum_max_digits(epoch):
    if epoch <= 15: return 3
    elif epoch <= 40: return 6
    else: return 10


VARIANTS = {
    # Baseline tied_proj (what we know works) — 372p
    'tp': dict(no_bias=False, ffn_dim=6, factored_pos=False, norm_type='ln',
               curriculum=False),

    # tied_proj + curriculum — 372p
    'tp_curr': dict(no_bias=False, ffn_dim=6, factored_pos=False, norm_type='ln',
                    curriculum=True),

    # tied_proj + no bias — 354p
    'tp_nb': dict(no_bias=True, ffn_dim=6, factored_pos=False, norm_type='bfln',
                  curriculum=False),

    # tied_proj + no bias + curriculum — 354p
    'tp_nb_curr': dict(no_bias=True, ffn_dim=6, factored_pos=False, norm_type='bfln',
                       curriculum=True),

    # tied_proj + no bias + FFN4 — 336p
    'tp_nb_f4': dict(no_bias=True, ffn_dim=4, factored_pos=False, norm_type='bfln',
                     curriculum=False),

    # tied_proj + no bias + FFN4 + curriculum — 336p
    'tp_nb_f4_curr': dict(no_bias=True, ffn_dim=4, factored_pos=False, norm_type='bfln',
                          curriculum=True),

    # tied_proj + no bias + FFN3 — 324p
    'tp_nb_f3': dict(no_bias=True, ffn_dim=3, factored_pos=False, norm_type='bfln',
                     curriculum=False),

    # tied_proj + no bias + FFN3 + curriculum — 324p
    'tp_nb_f3_curr': dict(no_bias=True, ffn_dim=3, factored_pos=False, norm_type='bfln',
                          curriculum=True),

    # tied_proj + factored pos — 321p
    'tp_fpos': dict(no_bias=False, ffn_dim=6, factored_pos=True, norm_type='ln',
                    curriculum=False),

    # tied_proj + factored pos + curriculum — 321p
    'tp_fpos_curr': dict(no_bias=False, ffn_dim=6, factored_pos=True, norm_type='ln',
                         curriculum=True),

    # tied_proj + no bias + factored pos — 303p
    'tp_nb_fpos': dict(no_bias=True, ffn_dim=6, factored_pos=True, norm_type='bfln',
                       curriculum=False),

    # tied_proj + no bias + FFN4 + factored pos — 285p
    'tp_nb_f4_fpos': dict(no_bias=True, ffn_dim=4, factored_pos=True, norm_type='bfln',
                          curriculum=False),

    # tied_proj + no bias + FFN4 + factored pos + curriculum — 285p
    'tp_nb_f4_fpos_curr': dict(no_bias=True, ffn_dim=4, factored_pos=True, norm_type='bfln',
                               curriculum=True),

    # tied_proj + no bias + FFN3 + factored pos — 273p
    'tp_nb_f3_fpos': dict(no_bias=True, ffn_dim=3, factored_pos=True, norm_type='bfln',
                          curriculum=False),

    # tied_proj + no bias + FFN3 + factored pos + curriculum — 273p
    'tp_nb_f3_fpos_curr': dict(no_bias=True, ffn_dim=3, factored_pos=True, norm_type='bfln',
                               curriculum=True),

    # HIGH PRIORITY: tied_proj + FFN4 (KEEP biases!) — ~346p
    'tp_f4': dict(no_bias=False, ffn_dim=4, factored_pos=False, norm_type='ln',
                  curriculum=False),

    # tied_proj + FFN3 (KEEP biases!) — ~333p
    'tp_f3': dict(no_bias=False, ffn_dim=3, factored_pos=False, norm_type='ln',
                  curriculum=False),

    # tied_proj + FFN4 + factored pos (keep biases) — ~295p
    'tp_f4_fpos': dict(no_bias=False, ffn_dim=4, factored_pos=True, norm_type='ln',
                       curriculum=False),

    # tied_proj + FFN3 + factored pos (keep biases) — ~282p
    'tp_f3_fpos': dict(no_bias=False, ffn_dim=3, factored_pos=True, norm_type='ln',
                       curriculum=False),
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
    cfg = VARIANTS[variant_name].copy()
    use_curriculum = cfg.pop('curriculum', False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Variant: {variant_name}")
    print(f"Config: {cfg}")
    print(f"Curriculum: {use_curriculum}")
    print(f"Device: {device}")

    model = WinnerModel(**cfg).to(device)
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
    save_name = f'best_winner_{variant_name}.pt'

    for epoch in range(1, 1001):
        if use_curriculum:
            max_d = get_curriculum_max_digits(epoch)
            train_dataset = CurriculumDataset(TRAIN_SAMPLES_PER_EPOCH, max_d, seed=None)
        else:
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
        curr_str = f" [max_d={max_d}]" if use_curriculum else ""
        print(f"Epoch {epoch:3d} | train_loss={epoch_loss/max(epoch_tokens,1):.4f} | "
              f"val_loss={val_loss:.4f} | da={val_da:.4f} | ea={val_ea:.4f} | "
              f"{bucket_str}{curr_str}")

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
    _, _, final_ea, final_bucket = evaluate(model,
        DataLoader(AdditionDataset(10000, seed=999), batch_size=BATCH_SIZE), device)
    bucket_str = " ".join(f"{nd}d:{acc:.3f}" for nd, acc in final_bucket.items())
    print(f"Final (10K, seed=999): ea={final_ea:.4f} | {bucket_str}")

    _, _, adder_ea, adder_bucket = evaluate(model,
        DataLoader(AdditionDataset(10000, seed=2025), batch_size=BATCH_SIZE), device)
    bucket_str = " ".join(f"{nd}d:{acc:.3f}" for nd, acc in adder_bucket.items())
    print(f"Final (10K, seed=2025): ea={adder_ea:.4f} | {bucket_str}")

    with open(f'results_winner_{variant_name}.json', 'w') as f:
        json.dump({'variant': variant_name, 'params': n_total, 'best_ea': best_exact_acc,
                   'best_epoch': best_epoch, 'final_ea_999': final_ea, 'final_ea_2025': adder_ea}, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, required=True, choices=list(VARIANTS.keys()))
    args = parser.parse_args()
    train(args.variant)


if __name__ == '__main__':
    main()
