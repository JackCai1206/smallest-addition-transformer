"""Minimal decoder-only transformer for integer addition.

Architecture:
  d_model = 6 = tok_dim (3) + pos_dim (3)
  First 3 dims: token embedding (hardcoded 3D spiral over digit value)
  Last 3 dims:  positional encoding (hardcoded 3D spiral over digit position)

  Attention (2 heads, split by dimension):
    Head 1: Q,K,V all from first 3 dims (token dims)
    Head 2: Q,K,V all from last 3 dims  (positional dims)
    Output projection maps concatenated heads back to d_model.

  No MLP / FFN — just attention + layernorm, repeated for N_LAYERS.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    VOCAB_SIZE, D_MODEL, TOK_DIM, POS_DIM, N_HEADS, HEAD_DIM,
    N_LAYERS, FFN_DIM, MAX_SEQ_LEN, MAX_DIGITS, ANSWER_LEN, N_POS,
    X_START, PLUS_POS, Y_START, EQ_POS, Z_START, EOS_POS,
    PLUS_TOKEN, EQUALS_TOKEN, EOS_TOKEN, PAD_TOKEN
)


def _spiral(index, period):
    """Return (cos, sin, linear) for a given index."""
    return (
        math.cos(2 * math.pi * index / period),
        math.sin(2 * math.pi * index / period),
        index / max(period - 1, 1),
    )


def build_token_embedding():
    """(VOCAB_SIZE, TOK_DIM=3) hardcoded 3D spiral over digit value 0-9."""
    emb = torch.zeros(VOCAB_SIZE, TOK_DIM)
    for d in range(10):
        emb[d, 0], emb[d, 1], emb[d, 2] = _spiral(d, 10)
    # Special tokens: distinct points
    emb[PLUS_TOKEN]   = torch.tensor([2.0, 0.0, -1.0])
    emb[EQUALS_TOKEN] = torch.tensor([0.0, 2.0, -1.0])
    emb[EOS_TOKEN]    = torch.tensor([-2.0, 0.0, -1.0])
    emb[PAD_TOKEN]    = torch.tensor([0.0, -2.0, -1.0])
    return emb


def build_positional_encoding():
    """(MAX_SEQ_LEN, POS_DIM=3) hardcoded 3D spiral over digit position 0-9.

    x[i], y[i], z[i] all get the same positional encoding for digit index i.
    Delimiters (+, =, EOS) and z[10] get distinct fixed values.
    """
    pe = torch.zeros(MAX_SEQ_LEN, POS_DIM)

    for i in range(MAX_DIGITS):
        pe[X_START + i, 0], pe[X_START + i, 1], pe[X_START + i, 2] = _spiral(i, N_POS)
    for i in range(MAX_DIGITS):
        pe[Y_START + i, 0], pe[Y_START + i, 1], pe[Y_START + i, 2] = _spiral(i, N_POS)
    for i in range(min(ANSWER_LEN, N_POS)):
        pe[Z_START + i, 0], pe[Z_START + i, 1], pe[Z_START + i, 2] = _spiral(i, N_POS)

    # z[10] (carry digit) — no matching operand position
    pe[Z_START + 10] = torch.tensor([0.0, 0.0, 1.5])

    # Delimiters
    pe[PLUS_POS] = torch.tensor([2.0, 0.0, -1.0])
    pe[EQ_POS]   = torch.tensor([0.0, 2.0, -1.0])
    pe[EOS_POS]  = torch.tensor([-2.0, 0.0, -1.0])

    return pe


class SplitHeadAttention(nn.Module):
    """Two-head attention where each head operates on different dims.

    Head 1: Q,K,V from first TOK_DIM (3) dims
    Head 2: Q,K,V from last POS_DIM (3) dims
    """

    def __init__(self):
        super().__init__()
        self.head_dim = HEAD_DIM  # 3

        # Head 1: token dims (first 3)
        self.q1 = nn.Linear(TOK_DIM, HEAD_DIM, bias=False)
        self.k1 = nn.Linear(TOK_DIM, HEAD_DIM, bias=False)
        self.v1 = nn.Linear(TOK_DIM, HEAD_DIM, bias=False)

        # Head 2: positional dims (last 3)
        self.q2 = nn.Linear(POS_DIM, HEAD_DIM, bias=False)
        self.k2 = nn.Linear(POS_DIM, HEAD_DIM, bias=False)
        self.v2 = nn.Linear(POS_DIM, HEAD_DIM, bias=False)

        # Output: concat of 2 heads (6) -> d_model (6)
        self.out_proj = nn.Linear(N_HEADS * HEAD_DIM, D_MODEL, bias=False)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN)).unsqueeze(0)
        )

    def _attend(self, q, k, v, T):
        """Single-head scaled dot-product attention with causal mask."""
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(self.causal_mask[:, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        return attn @ v

    def forward(self, x):
        B, T, _ = x.shape
        x_tok = x[:, :, :TOK_DIM]   # (B, T, 3)
        x_pos = x[:, :, TOK_DIM:]   # (B, T, 3)

        # Head 1: token dims
        out1 = self._attend(self.q1(x_tok), self.k1(x_tok), self.v1(x_tok), T)
        # Head 2: positional dims
        out2 = self._attend(self.q2(x_pos), self.k2(x_pos), self.v2(x_pos), T)

        out = torch.cat([out1, out2], dim=-1)  # (B, T, 6)
        return self.out_proj(out)


class AttentionBlock(nn.Module):
    """Attention + LayerNorm, no MLP."""
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(D_MODEL)
        self.attn = SplitHeadAttention()

    def forward(self, x):
        return x + self.attn(self.ln(x))


class SmallestAdditionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # Both embeddings are hardcoded buffers (0 learnable params)
        self.register_buffer("tok_emb", build_token_embedding())
        self.register_buffer("pos_enc", build_positional_encoding())

        self.blocks = nn.ModuleList([AttentionBlock() for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, idx):
        B, T = idx.shape
        tok = self.tok_emb[idx]                              # (B, T, 3)
        pos = self.pos_enc[:T].unsqueeze(0).expand(B, -1, -1)  # (B, T, 3)
        x = torch.cat([tok, pos], dim=-1)                    # (B, T, 6)

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
