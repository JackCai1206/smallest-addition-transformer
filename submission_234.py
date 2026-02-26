"""
Submission for the Nano Transformer Adder leaderboard.

234-parameter transformer that learns 10-digit addition.

Architecture: 1-layer decoder, d_model=6, 2 heads, ffn=2, tied output head
  - First 3 dims: token embedding (spiral-initialized, trainable)
  - Last 3 dims:  parametric spiral positional encoding (4 learned params)
  - Q,K from positional dims only; V from token dims only
  - Tied output: Linear(6->3) @ tok_emb.T instead of Linear(6->14)
  - Shared XYZ: X[i]=Y[i]=Z[i] with parametric spiral saves 86 params
  - FFN dim=2 (minimal) + no weight decay
  - Digits in LSB-first order for natural carry propagation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# -- Constants --
VOCAB_SIZE = 14
D_MODEL = 6
TOK_DIM = 3
POS_DIM = 3
N_HEADS = 2
HEAD_DIM = 3
FFN_DIM = 2
MAX_SEQ_LEN = 34
MAX_DIGITS = 10
ANSWER_LEN = 11

PLUS_TOKEN = 10
EQUALS_TOKEN = 11
EOS_TOKEN = 12
PAD_TOKEN = 13


# -- Spiral initialization --
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


def build_pos_index_map():
    sources = []
    indices = []
    for p in range(MAX_SEQ_LEN):
        if p < 10:
            sources.append(0); indices.append(p)
        elif p == 10:
            sources.append(2); indices.append(0)
        elif p < 21:
            sources.append(0); indices.append(p - 11)
        elif p == 21:
            sources.append(2); indices.append(1)
        elif p < 33:
            z_idx = p - 22
            if z_idx < 10:
                sources.append(0); indices.append(z_idx)
            else:
                sources.append(1); indices.append(0)
        else:
            sources.append(2); indices.append(2)
    return torch.tensor(sources), torch.tensor(indices)


# -- Model --
class SmallestAdditionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, TOK_DIM)
        self.tok_emb.weight.data.copy_(build_token_embedding())

        # Parametric spiral positional encoding: 4 params generate digit positions
        self.pos_spiral_amp = nn.Parameter(torch.tensor(1.0))
        self.pos_spiral_phase = nn.Parameter(torch.tensor(0.0))
        self.pos_spiral_slope = nn.Parameter(torch.tensor(1.0/9.0))
        self.pos_spiral_offset = nn.Parameter(torch.tensor(0.0))
        self.z10_enc = nn.Parameter(torch.zeros(1, POS_DIM))
        self.special_enc = nn.Parameter(torch.zeros(3, POS_DIM))

        sources, indices = build_pos_index_map()
        self.register_buffer('_pos_sources', sources)
        self.register_buffer('_pos_indices', indices)

        self.q_proj = nn.Linear(POS_DIM, N_HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(POS_DIM, N_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(TOK_DIM, N_HEADS * HEAD_DIM, bias=False)
        self.out_proj = nn.Linear(N_HEADS * HEAD_DIM, D_MODEL, bias=False)

        self.ln1 = nn.LayerNorm(D_MODEL)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.ln_f = nn.LayerNorm(D_MODEL)

        self.ffn = nn.Sequential(
            nn.Linear(D_MODEL, FFN_DIM),
            nn.GELU(),
            nn.Linear(FFN_DIM, D_MODEL),
        )

        self.head_proj = nn.Linear(D_MODEL, TOK_DIM, bias=False)

        self.register_buffer("causal_mask",
            torch.tril(torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN)).unsqueeze(0).unsqueeze(0))

    def _get_digit_pos_enc(self):
        """Generate digit position encodings from 4 spiral parameters."""
        i = torch.arange(10, dtype=torch.float, device=self.pos_spiral_amp.device)
        enc = torch.zeros(10, 3, device=self.pos_spiral_amp.device)
        angle = 2 * math.pi * i / 10 + self.pos_spiral_phase
        enc[:, 0] = self.pos_spiral_amp * torch.cos(angle)
        enc[:, 1] = self.pos_spiral_amp * torch.sin(angle)
        enc[:, 2] = self.pos_spiral_slope * i + self.pos_spiral_offset
        return enc

    def _get_pos(self, T):
        digit_enc = self._get_digit_pos_enc()
        pos = torch.zeros(T, POS_DIM, device=digit_enc.device)
        for t in range(T):
            src = self._pos_sources[t].item()
            idx = self._pos_indices[t].item()
            if src == 0:
                pos[t] = digit_enc[idx]
            elif src == 1:
                pos[t] = self.z10_enc[idx]
            else:
                pos[t] = self.special_enc[idx]
        return pos

    def forward(self, idx):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self._get_pos(T).unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([tok, pos], dim=-1)

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

        x = x + self.ffn(self.ln2(x))
        x = self.ln_f(x)
        return self.head_proj(x) @ self.tok_emb.weight.T

    @torch.no_grad()
    def generate(self, prompt):
        self.eval()
        B, T_prompt = prompt.shape
        full_seq = torch.zeros(B, T_prompt + ANSWER_LEN + 1, dtype=torch.long, device=prompt.device)
        full_seq[:, :T_prompt] = prompt
        for step in range(ANSWER_LEN + 1):
            T = T_prompt + step
            logits = self.forward(full_seq[:, :T])
            next_token = logits[:, -1].argmax(dim=-1)
            full_seq[:, T] = next_token
        return full_seq[:, T_prompt:]


# -- Submission interface --
def build_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SmallestAdditionTransformer()

    import os
    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model_234.pt")
    state = torch.load(ckpt_path, weights_only=True, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())

    metadata = {
        "name": "Spiral Addition Transformer (parametric-pos, shared-XYZ, tied-proj, FFN2)",
        "author": "Jack Cai",
        "params": n_params,
        "architecture": "1L decoder, d=6 (3 tok + 3 pos), 2h, hd=3, ff=2, parametric spiral pos, shared XYZ, tied output",
        "tricks": [
            "Spiral-initialized trainable token embeddings: (cos,sin,linear)",
            "Parametric spiral positional encoding: 4 params generate all digit positions",
            "Split-head attention: Q,K from positional dims; V from token dims",
            "Shared XYZ positional encoding: X[i]=Y[i]=Z[i] from parametric spiral",
            "Tied output head: Linear(6->3) @ tok_emb.T saves 66 params",
            "FFN dim=2 (minimal), no weight decay",
            "LSB-first digit ordering for natural carry propagation",
        ],
    }
    return model, metadata


def add(model, a: int, b: int) -> int:
    device = next(model.parameters()).device

    x_digits = [(a // 10**i) % 10 for i in range(MAX_DIGITS)]
    y_digits = [(b // 10**i) % 10 for i in range(MAX_DIGITS)]

    prompt = x_digits + [PLUS_TOKEN] + y_digits + [EQUALS_TOKEN]
    prompt_tensor = torch.tensor([prompt], dtype=torch.long, device=device)

    with torch.no_grad():
        generated = model.generate(prompt_tensor)

    result = 0
    gen = generated[0].tolist()
    for i, tok in enumerate(gen):
        if tok == EOS_TOKEN or tok >= 10:
            break
        result += tok * (10 ** i)

    return result
