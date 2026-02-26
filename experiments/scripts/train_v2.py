"""Next-generation model: incorporating leaderboard techniques.

Key insights from AdderBoard leaders:
1. Rank-3 factorization on all linear layers: W = A(m,r) × B(r,n)
2. Tied embeddings: input tok_emb = output head (transposed)
3. Shared-A tied-KV: Q = x×A×Bq, K=V=x×A×Bkv
4. Curriculum learning: 1-3 → 1-6 → 1-10 digits
5. d_model=4 works with aggressive factorization

We test multiple approaches:
- v2_baseline: Our 438p model + curriculum learning
- v2_d4: d=4, tied emb, curriculum, spiral init, long training
- v2_d4_r3: d=4, rank-3 factorization, tied emb, curriculum
- v2_d6_r3: d=6, our architecture, rank-3 factorization on head+pos
- v2_d4_shareA: d=4, shared-A tied-KV, tied emb
"""

import argparse, math, os, json
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict
import numpy as np
from config import (
    VOCAB_SIZE, MAX_SEQ_LEN, MAX_DIGITS, ANSWER_LEN,
    EQ_POS, BATCH_SIZE, LR,
    PLUS_TOKEN, EQUALS_TOKEN, EOS_TOKEN, PAD_TOKEN
)
from data import AdditionDataset


# ── Curriculum data generation ─────────────────────────────────────────
def generate_curriculum_batch(num_samples, max_digits, rng):
    """Generate addition examples with digits up to max_digits."""
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
    tokens[:, MAX_DIGITS] = PLUS_TOKEN
    tmp = y.copy()
    for d in range(MAX_DIGITS):
        tokens[:, MAX_DIGITS + 1 + d] = tmp % 10; tmp //= 10
    tokens[:, 2 * MAX_DIGITS + 1] = EQUALS_TOKEN
    tmp = z.copy()
    for d in range(MAX_DIGITS + 1):
        tokens[:, 2 * MAX_DIGITS + 2 + d] = tmp % 10; tmp //= 10
    tokens[:, -1] = EOS_TOKEN

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


# ── Building blocks ──────────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return self.scale * x / torch.sqrt(torch.mean(x**2, -1, keepdim=True) + self.eps)


class BiasFreeLN(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return self.weight * (x - mean) / torch.sqrt(var + self.eps)


class FactoredLinear(nn.Module):
    """Low-rank factored linear: y = x @ A @ B, where A=(in,r), B=(r,out)."""
    def __init__(self, in_dim, out_dim, rank=3):
        super().__init__()
        self.A = nn.Parameter(torch.randn(in_dim, rank) * 0.1)
        self.B = nn.Parameter(torch.randn(rank, out_dim) * 0.1)
    def forward(self, x):
        return x @ self.A @ self.B


# ── Spiral initialization helpers ──────────────────────────────────────
def _spiral(index, period):
    return (math.cos(2*math.pi*index/period),
            math.sin(2*math.pi*index/period),
            index/max(period-1, 1))


def build_tok_emb_4d():
    """4D token embedding using double spiral."""
    emb = torch.zeros(VOCAB_SIZE, 4)
    for d in range(10):
        c1, s1, _ = _spiral(d, 10)
        c2, s2, _ = _spiral(d, 5)  # second harmonic
        emb[d] = torch.tensor([c1, s1, c2, s2])
    emb[PLUS_TOKEN]   = torch.tensor([2.0, 0.0, -1.0, 0.0])
    emb[EQUALS_TOKEN] = torch.tensor([0.0, 2.0, 0.0, -1.0])
    emb[EOS_TOKEN]    = torch.tensor([-2.0, 0.0, 1.0, 0.0])
    emb[PAD_TOKEN]    = torch.tensor([0.0, -2.0, 0.0, 1.0])
    return emb


def build_pos_enc_4d():
    """4D positional encoding using double spiral."""
    pe = torch.zeros(MAX_SEQ_LEN, 4)
    for i in range(MAX_DIGITS):
        c1, s1, _ = _spiral(i, 10)
        c2, s2, _ = _spiral(i, 20)
        pe[i] = torch.tensor([c1, s1, c2, s2])        # X digits
        pe[11+i] = torch.tensor([c1, s1, c2, s2])      # Y digits
    for i in range(MAX_DIGITS+1):
        if i < 10:
            c1, s1, _ = _spiral(i, 10)
            c2, s2, _ = _spiral(i, 20)
        else:
            c1, s1, c2, s2 = 0, 0, 0, 1.5
        pe[22+i] = torch.tensor([c1, s1, c2, s2])      # Z digits
    pe[10] = torch.tensor([2.0, 0.0, -1.0, 0.0])       # +
    pe[21] = torch.tensor([0.0, 2.0, 0.0, -1.0])       # =
    pe[33] = torch.tensor([-2.0, 0.0, 1.0, 0.0])       # EOS
    return pe


def build_tok_emb_3d():
    """3D token embedding (our original spiral)."""
    emb = torch.zeros(VOCAB_SIZE, 3)
    for d in range(10):
        emb[d, 0], emb[d, 1], emb[d, 2] = _spiral(d, 10)
    emb[PLUS_TOKEN]   = torch.tensor([2.0, 0.0, -1.0])
    emb[EQUALS_TOKEN] = torch.tensor([0.0, 2.0, -1.0])
    emb[EOS_TOKEN]    = torch.tensor([-2.0, 0.0, -1.0])
    emb[PAD_TOKEN]    = torch.tensor([0.0, -2.0, -1.0])
    return emb


def build_pos_enc_3d():
    """3D positional encoding (our original spiral)."""
    pe = torch.zeros(MAX_SEQ_LEN, 3)
    for i in range(MAX_DIGITS):
        pe[i, 0], pe[i, 1], pe[i, 2] = _spiral(i, 10)
        pe[11+i, 0], pe[11+i, 1], pe[11+i, 2] = _spiral(i, 10)
    for i in range(min(MAX_DIGITS+1, 10)):
        pe[22+i, 0], pe[22+i, 1], pe[22+i, 2] = _spiral(i, 10)
    pe[32] = torch.tensor([0.0, 0.0, 1.5])
    pe[10] = torch.tensor([2.0, 0.0, -1.0])
    pe[21] = torch.tensor([0.0, 2.0, -1.0])
    pe[33] = torch.tensor([-2.0, 0.0, -1.0])
    return pe


# ── Model: d=4 with tied embeddings ──────────────────────────────────
class ModelD4(nn.Module):
    """d=4, 1 head, tied embeddings, standard attention (no split dims)."""
    def __init__(self, ffn_dim=8, use_rank3=False, shared_a_rank=0,
                 norm_type='bfln', factored_pos_rank=0):
        super().__init__()
        self.d = 4
        self.use_rank3 = use_rank3
        self.shared_a_rank = shared_a_rank

        # Token embedding (tied with output)
        self.tok_emb = nn.Embedding(VOCAB_SIZE, 4)
        self.tok_emb.weight.data.copy_(build_tok_emb_4d())

        # Position encoding
        if factored_pos_rank > 0:
            r = factored_pos_rank
            self.pos_A = nn.Parameter(torch.zeros(MAX_SEQ_LEN, r))
            self.pos_B = nn.Parameter(torch.zeros(r, 4))
            # Initialize from full pos encoding via SVD-like factorization
            full_pe = build_pos_enc_4d()
            U, S, Vh = torch.linalg.svd(full_pe, full_matrices=False)
            self.pos_A.data.copy_(U[:, :r] * S[:r].unsqueeze(0))
            self.pos_B.data.copy_(Vh[:r])
        else:
            self.pos_enc = nn.Parameter(build_pos_enc_4d())

        self.factored_pos_rank = factored_pos_rank

        # Attention
        if shared_a_rank > 0:
            # Shared-A tied-KV: Q = x@A@Bq, K=V=x@A@Bkv
            r = shared_a_rank
            self.shared_A = nn.Parameter(torch.randn(4, r) * 0.1)
            self.Bq = nn.Parameter(torch.randn(r, 4) * 0.1)
            self.Bkv = nn.Parameter(torch.randn(r, 4) * 0.1)
        elif use_rank3:
            self.q_proj = FactoredLinear(4, 4, rank=3)
            self.k_proj = FactoredLinear(4, 4, rank=3)
            self.v_proj = FactoredLinear(4, 4, rank=3)
        else:
            self.q_proj = nn.Linear(4, 4, bias=False)
            self.k_proj = nn.Linear(4, 4, bias=False)
            self.v_proj = nn.Linear(4, 4, bias=False)

        if use_rank3:
            self.out_proj = FactoredLinear(4, 4, rank=2)
        else:
            self.out_proj = nn.Linear(4, 4, bias=False)

        # Norms
        NormClass = RMSNorm if norm_type == 'rms' else BiasFreeLN
        self.ln1 = NormClass(4)
        self.ln2 = NormClass(4)
        self.ln_f = NormClass(4)

        # FFN
        if use_rank3:
            self.ffn_up = FactoredLinear(4, ffn_dim, rank=3)
            self.ffn_down = FactoredLinear(ffn_dim, 4, rank=3)
        else:
            self.ffn_up = nn.Linear(4, ffn_dim, bias=False)
            self.ffn_down = nn.Linear(ffn_dim, 4, bias=False)

        self.register_buffer("causal_mask",
            torch.tril(torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN)).unsqueeze(0).unsqueeze(0))

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'tok_emb' in name or 'pos_enc' in name or 'pos_A' in name or 'pos_B' in name:
                continue
            if 'scale' in name or 'weight' in name and p.dim() == 1:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _get_pos(self, T):
        if self.factored_pos_rank > 0:
            return (self.pos_A[:T] @ self.pos_B)
        return self.pos_enc[:T]

    def forward(self, idx):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self._get_pos(T).unsqueeze(0).expand(B, -1, -1)
        x = tok + pos  # additive (not concat!) so tok_emb dim = d_model

        # Pre-norm attention
        h = self.ln1(x)
        if self.shared_a_rank > 0:
            shared = h @ self.shared_A
            q = shared @ self.Bq
            k = v = shared @ self.Bkv
        else:
            q = self.q_proj(h)
            k = self.k_proj(h)
            v = self.v_proj(h)

        # Single-head attention
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d)
        mask = self.causal_mask[0, 0, :T, :T]  # (T, T) — squeeze head dims for single-head
        attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        x = x + self.out_proj(attn @ v)

        # Pre-norm FFN
        h = self.ln2(x)
        x = x + self.ffn_down(F.gelu(self.ffn_up(h)))

        x = self.ln_f(x)
        # Tied output head: logits = x @ tok_emb.T
        return x @ self.tok_emb.weight.T

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


# ── Model: d=6 with split dims (our architecture) + rank-3 head ──────
class ModelD6R3(nn.Module):
    """Our split-dim architecture with rank-factored output head and/or pos encoding."""
    def __init__(self, head_rank=3, factored_pos_rank=0, no_bias=True,
                 ffn_dim=6, norm_type='bfln'):
        super().__init__()
        self.d = 6
        self.tok_dim = 3
        self.pos_dim = 3
        self.n_heads = 2
        self.head_dim = 3
        self.head_rank = head_rank

        self.tok_emb = nn.Embedding(VOCAB_SIZE, 3)
        self.tok_emb.weight.data.copy_(build_tok_emb_3d())

        if factored_pos_rank > 0:
            r = factored_pos_rank
            full_pe = build_pos_enc_3d()
            U, S, Vh = torch.linalg.svd(full_pe, full_matrices=False)
            self.pos_A = nn.Parameter(U[:, :r] * S[:r].unsqueeze(0))
            self.pos_B = nn.Parameter(Vh[:r])
        else:
            self.pos_enc = nn.Parameter(build_pos_enc_3d())
        self.factored_pos_rank = factored_pos_rank

        # Attention (QK from pos, V from tok)
        self.q_proj = nn.Linear(3, 6, bias=False)
        self.k_proj = nn.Linear(3, 6, bias=False)
        self.v_proj = nn.Linear(3, 6, bias=False)
        self.out_proj = nn.Linear(6, 6, bias=False)

        NormClass = RMSNorm if norm_type == 'rms' else (BiasFreeLN if no_bias else nn.LayerNorm)
        self.ln1 = NormClass(6)
        self.ln2 = NormClass(6)
        self.ln_f = NormClass(6)

        if no_bias:
            self.ffn = nn.Sequential(
                nn.Linear(6, ffn_dim, bias=False), nn.GELU(), nn.Linear(ffn_dim, 6, bias=False))
        else:
            self.ffn = nn.Sequential(
                nn.Linear(6, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, 6))

        if head_rank > 0:
            self.head_A = nn.Linear(6, head_rank, bias=False)
            self.head_B = nn.Linear(head_rank, VOCAB_SIZE, bias=False)
        else:
            self.head = nn.Linear(6, VOCAB_SIZE, bias=False)

        self.register_buffer("causal_mask",
            torch.tril(torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN)).unsqueeze(0).unsqueeze(0))
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'tok_emb' in name or 'pos_enc' in name or 'pos_A' in name or 'pos_B' in name:
                continue
            if 'scale' in name or ('weight' in name and p.dim() == 1):
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _get_pos(self, T):
        if self.factored_pos_rank > 0:
            return (self.pos_A[:T] @ self.pos_B)
        return self.pos_enc[:T]

    def forward(self, idx):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self._get_pos(T).unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([tok, pos], dim=-1)

        h = self.ln1(x)
        x_pos = h[:, :, 3:]
        x_tok = h[:, :, :3]
        q = self.q_proj(x_pos).view(B, T, 2, 3).transpose(1, 2)
        k = self.k_proj(x_pos).view(B, T, 2, 3).transpose(1, 2)
        v = self.v_proj(x_tok).view(B, T, 2, 3).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(3)
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, 6)
        x = x + self.out_proj(out)

        x = x + self.ffn(self.ln2(x))
        x = self.ln_f(x)

        if self.head_rank > 0:
            return self.head_B(self.head_A(x))
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


# ── Variant configurations ─────────────────────────────────────────────
VARIANTS = {
    # Our d=6 architecture with curriculum learning
    'baseline_curr': dict(
        model_type='d6', head_rank=0, factored_pos_rank=0, no_bias=False,
        ffn_dim=6, norm_type='ln', curriculum=True, n_steps=200000),

    # Our d=6 with no-bias LN + rank-3 factored head + curriculum
    'd6_nb_fh3': dict(
        model_type='d6', head_rank=3, factored_pos_rank=0, no_bias=True,
        ffn_dim=6, norm_type='bfln', curriculum=True, n_steps=200000),

    # d=6, no-bias, rank-2 factored head, FFN=4, curriculum
    'd6_shrink': dict(
        model_type='d6', head_rank=2, factored_pos_rank=0, no_bias=True,
        ffn_dim=4, norm_type='bfln', curriculum=True, n_steps=200000),

    # d=4, tied embed, standard attn, curriculum, long training
    'd4_basic': dict(
        model_type='d4', ffn_dim=8, use_rank3=False, shared_a_rank=0,
        norm_type='bfln', factored_pos_rank=0, curriculum=True, n_steps=200000),

    # d=4, tied embed, shared-A tied-KV (r=2)
    'd4_shareA': dict(
        model_type='d4', ffn_dim=8, use_rank3=False, shared_a_rank=2,
        norm_type='bfln', factored_pos_rank=0, curriculum=True, n_steps=200000),

    # d=4, rank-3 factorization everywhere
    'd4_r3': dict(
        model_type='d4', ffn_dim=8, use_rank3=True, shared_a_rank=0,
        norm_type='bfln', factored_pos_rank=3, curriculum=True, n_steps=200000),

    # d=4, shared-A + factored pos
    'd4_shareA_fpos': dict(
        model_type='d4', ffn_dim=8, use_rank3=False, shared_a_rank=2,
        norm_type='bfln', factored_pos_rank=3, curriculum=True, n_steps=200000),

    # d=4, RMSNorm (to match leader)
    'd4_rms': dict(
        model_type='d4', ffn_dim=8, use_rank3=False, shared_a_rank=0,
        norm_type='rms', factored_pos_rank=0, curriculum=True, n_steps=200000),

    # d=4, everything: shareA + factored pos + rms
    'd4_full': dict(
        model_type='d4', ffn_dim=8, use_rank3=False, shared_a_rank=2,
        norm_type='rms', factored_pos_rank=3, curriculum=True, n_steps=200000),

    # d=4, shared-A rank 3 (matching leader's architecture more closely)
    'd4_shareA3': dict(
        model_type='d4', ffn_dim=8, use_rank3=False, shared_a_rank=3,
        norm_type='bfln', factored_pos_rank=0, curriculum=True, n_steps=200000),

    # d=4, grokking: NO curriculum, just long training (like the 311p leader)
    'd4_grok': dict(
        model_type='d4', ffn_dim=8, use_rank3=False, shared_a_rank=0,
        norm_type='bfln', factored_pos_rank=0, curriculum=False, n_steps=500000),

    # d=4, shared-A3 + factored pos + rms (closest to 311p leader)
    'd4_leader': dict(
        model_type='d4', ffn_dim=8, use_rank3=False, shared_a_rank=3,
        norm_type='rms', factored_pos_rank=3, curriculum=True, n_steps=200000),
}


# ── Evaluation ──────────────────────────────────────────────────────────
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


# ── Training with curriculum ───────────────────────────────────────────
def get_curriculum_max_digits(epoch):
    """3-phase curriculum: easier digits first, gradually increase.
    Phase 1 (epochs 1-15): 1-3 digit operands
    Phase 2 (epochs 16-40): 1-6 digit operands
    Phase 3 (epochs 41+): 1-10 digit operands (full range)
    """
    if epoch <= 15:
        return 3
    elif epoch <= 40:
        return 6
    else:
        return 10


def train(variant_name):
    cfg = VARIANTS[variant_name]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Variant: {variant_name}")
    print(f"Config: {cfg}")
    print(f"Device: {device}")

    # Build model
    model_type = cfg.pop('model_type')
    use_curriculum = cfg.pop('curriculum', False)
    n_steps = cfg.pop('n_steps', 200000)

    if model_type == 'd4':
        model = ModelD4(**{k: v for k, v in cfg.items()
                         if k in ('ffn_dim', 'use_rank3', 'shared_a_rank',
                                  'norm_type', 'factored_pos_rank')}).to(device)
    else:
        model = ModelD6R3(**{k: v for k, v in cfg.items()
                           if k in ('head_rank', 'factored_pos_rank', 'no_bias',
                                    'ffn_dim', 'norm_type')}).to(device)

    n_trainable, n_total = model.count_parameters()
    print(f"Parameters: {n_trainable} trainable / {n_total} total")
    for name, p in model.named_parameters():
        print(f"  {name}: {list(p.shape)} = {p.numel()}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    val_dataset = AdditionDataset(2048, seed=42)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    best_exact_acc = -1.0
    best_epoch = 0
    patience_counter = 0
    patience = 500
    save_name = f'best_v2_{variant_name}.pt'
    max_epochs = 1000  # plenty of time; early stopping with patience=500

    for epoch in range(1, max_epochs + 1):
        if use_curriculum:
            max_d = get_curriculum_max_digits(epoch)
            train_dataset = CurriculumDataset(50000, max_d, seed=None)
        else:
            train_dataset = AdditionDataset(50000, seed=None)
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

        # Evaluate on full 1-10 digit val set
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
    print(f"Final (10K): ea={final_ea:.4f} | {bucket_str}")

    with open(f'results_v2_{variant_name}.json', 'w') as f:
        json.dump({'variant': variant_name, 'params': n_total, 'best_ea': best_exact_acc,
                   'best_epoch': best_epoch, 'final_ea': final_ea}, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, required=True, choices=list(VARIANTS.keys()))
    args = parser.parse_args()
    train(args.variant)


if __name__ == '__main__':
    main()
