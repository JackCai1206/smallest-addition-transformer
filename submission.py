"""
Submission for the Nano Transformer Adder leaderboard.

438-parameter transformer that learns 10-digit addition in ~75 epochs
with no grokking, thanks to structured spiral initialization of embeddings.

Architecture: 1-layer decoder, d_model=6, 2 heads, ffn=6
  - First 3 dims: token embedding (spiral-initialized, trainable)
  - Last 3 dims:  positional encoding (spiral-initialized, trainable)
  - Q,K from positional dims only; V from token dims only
  - Digits in LSB-first order for natural carry propagation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ── Constants ──────────────────────────────────────────────────────────────
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
EQ_POS = 21

PLUS_TOKEN = 10
EQUALS_TOKEN = 11
EOS_TOKEN = 12
PAD_TOKEN = 13


# ── Spiral initialization ─────────────────────────────────────────────────
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
    N_POS = 10
    pe = torch.zeros(MAX_SEQ_LEN, POS_DIM)
    for i in range(MAX_DIGITS):
        pe[i, 0], pe[i, 1], pe[i, 2] = _spiral(i, N_POS)           # x digits
    for i in range(MAX_DIGITS):
        pe[11 + i, 0], pe[11 + i, 1], pe[11 + i, 2] = _spiral(i, N_POS)  # y digits
    for i in range(min(ANSWER_LEN, N_POS)):
        pe[22 + i, 0], pe[22 + i, 1], pe[22 + i, 2] = _spiral(i, N_POS)  # z digits
    pe[32] = torch.tensor([0.0, 0.0, 1.5])   # z[10] carry
    pe[10] = torch.tensor([2.0, 0.0, -1.0])  # +
    pe[21] = torch.tensor([0.0, 2.0, -1.0])  # =
    pe[33] = torch.tensor([-2.0, 0.0, -1.0]) # EOS
    return pe


# ── Model ──────────────────────────────────────────────────────────────────
class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_heads = N_HEADS
        self.head_dim = HEAD_DIM
        self.q_proj = nn.Linear(POS_DIM, N_HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(POS_DIM, N_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(TOK_DIM, N_HEADS * HEAD_DIM, bias=False)
        self.out_proj = nn.Linear(N_HEADS * HEAD_DIM, D_MODEL, bias=False)
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


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = CausalSelfAttention()
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


class SmallestAdditionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, TOK_DIM)
        self.pos_enc = nn.Parameter(torch.zeros(MAX_SEQ_LEN, POS_DIM))
        self.blocks = nn.ModuleList([TransformerBlock()])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)

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
        B, T_prompt = prompt.shape
        # Pre-allocate full sequence to avoid repeated concat
        full_seq = torch.zeros(B, T_prompt + ANSWER_LEN + 1, dtype=torch.long, device=prompt.device)
        full_seq[:, :T_prompt] = prompt
        for step in range(ANSWER_LEN + 1):
            T = T_prompt + step
            logits = self.forward(full_seq[:, :T])
            next_token = logits[:, -1].argmax(dim=-1)
            full_seq[:, T] = next_token
        return full_seq[:, T_prompt:]


# ── Submission interface ───────────────────────────────────────────────────
def build_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SmallestAdditionTransformer()

    # Load trained weights
    import os
    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model_trainable_emb.pt")
    state = torch.load(ckpt_path, weights_only=True, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())

    metadata = {
        "name": "Spiral Addition Transformer",
        "author": "Jack Cai",
        "params": n_params,
        "architecture": "1L decoder, d=6 (3 tok + 3 pos), 2h, hd=3, ff=6",
        "tricks": [
            "Spiral-initialized trainable embeddings (no grokking needed)",
            "Q,K from positional dims only; V from token dims only",
            "LSB-first digit ordering for natural carry propagation",
            "Converges in ~75 epochs (~3.75M samples) vs 35K+ steps for grokking approaches",
        ],
    }
    return model, metadata


def add(model, a: int, b: int) -> int:
    device = next(model.parameters()).device

    # Encode inputs as LSB-first digit sequences, padded to 10 digits
    x_digits = [(a // 10**i) % 10 for i in range(MAX_DIGITS)]
    y_digits = [(b // 10**i) % 10 for i in range(MAX_DIGITS)]

    prompt = x_digits + [PLUS_TOKEN] + y_digits + [EQUALS_TOKEN]
    prompt_tensor = torch.tensor([prompt], dtype=torch.long, device=device)

    with torch.no_grad():
        generated = model.generate(prompt_tensor)  # (1, ANSWER_LEN+1)

    # Decode: LSB-first digits, stop at EOS
    result = 0
    gen = generated[0].tolist()
    for i, tok in enumerate(gen):
        if tok == EOS_TOKEN or tok >= 10:
            break
        result += tok * (10 ** i)

    return result
