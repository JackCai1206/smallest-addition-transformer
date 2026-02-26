"""Variant: 2 layers, 2 heads, no MLP.

Q,K always from last 3 dims (positional), V always from first 3 dims (token).
Same as the original single-layer design but with 2 layers and no FFN.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    VOCAB_SIZE, D_MODEL, TOK_DIM, POS_DIM, N_HEADS, HEAD_DIM,
    N_LAYERS, MAX_SEQ_LEN, ANSWER_LEN,
    PLUS_TOKEN, EQUALS_TOKEN, EOS_TOKEN, PAD_TOKEN
)
from model import build_token_embedding, build_positional_encoding


class QKPosVTokAttention(nn.Module):
    """Both heads: Q,K from pos dims (last 3), V from tok dims (first 3)."""

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


class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(D_MODEL)
        self.attn = QKPosVTokAttention()

    def forward(self, x):
        return x + self.attn(self.ln(x))


class SmallestAdditionTransformerQKPosVTok(nn.Module):
    def __init__(self):
        super().__init__()
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
        tok = self.tok_emb[idx]
        pos = self.pos_enc[:T].unsqueeze(0).expand(B, -1, -1)
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
