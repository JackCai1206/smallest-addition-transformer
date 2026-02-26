"""d5 experiments with shared position encodings.

Key insight: In our split-dim architecture (QK from pos, V from tok),
X[i] and Y[i] can share position encodings because:
1. Q/K projections are learned — the model attends based on position similarity
2. V reads from token dims — so the VALUES for X[i] and Y[i] are different
3. For addition: Z[i] should attend equally to X[i] and Y[i], which is natural
   when they share position encoding

Shared-XY pos: digit_enc(10,3) + z_enc(11,3) + special(3,3) = 72 params (vs 102)
Shared-XYZ pos: digit_enc(10,3) + z10(1,3) + special(3,3) = 42 params (vs 102)
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
from train_d5 import build_tok_emb_2d, _spiral3, _spiral2, evaluate


class RMSNorm(nn.Module):
    """RMSNorm: like LayerNorm but no mean centering, no bias. Saves 50% params."""
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def build_pos_enc_full_3d():
    """Full (34, 3) PE."""
    from model import build_positional_encoding
    return build_positional_encoding()


class D5SharedPosModel(nn.Module):
    """D5 model with shared position encodings between X/Y (and optionally Z)."""
    def __init__(self, tok_dim=2, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=3,
                 shared_pos='xy', ffn_bias=True, tied_ffn=False, tied_qk=False,
                 shared_ln=False, out_proj_dim=None, param_spiral=False,
                 norm_type='ln', tied_vo=False, param_spiral_pos=False,
                 shared_ln_f=False):
        super().__init__()
        self.tok_dim = tok_dim
        self.pos_dim = pos_dim
        self.d_model = tok_dim + pos_dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.attn_dim = n_heads * head_dim
        self.shared_pos = shared_pos
        self.param_spiral = param_spiral
        self.norm_type = norm_type
        self.tied_vo = tied_vo
        self.param_spiral_pos = param_spiral_pos
        self.shared_ln_f = shared_ln_f

        # Token embedding
        if param_spiral and tok_dim == 3:
            # Parametric spiral: 5 params for digits + 4×3 for special tokens
            self.spiral_amp = nn.Parameter(torch.tensor(1.0))
            self.spiral_phase = nn.Parameter(torch.tensor(0.0))
            self.spiral_slope = nn.Parameter(torch.tensor(1.0/9.0))
            self.spiral_offset = nn.Parameter(torch.tensor(0.0))
            self.special_tok_emb = nn.Parameter(torch.zeros(4, 3))
            # Initialize special tokens from full embedding
            from model import build_token_embedding
            full_emb = build_token_embedding()
            self.special_tok_emb.data.copy_(full_emb[10:14])  # +, =, EOS, PAD
        else:
            self.tok_emb = nn.Embedding(VOCAB_SIZE, tok_dim)
            if tok_dim == 2:
                self.tok_emb.weight.data.copy_(build_tok_emb_2d())
            elif tok_dim == 3:
                from model import build_token_embedding
                self.tok_emb.weight.data.copy_(build_token_embedding())

        # Position encoding (shared)
        if param_spiral_pos and shared_pos == 'xyz' and pos_dim == 3:
            # Parametric spiral for digit positions: 4 params + z10(3) + special(9) = 16
            self.pos_spiral_amp = nn.Parameter(torch.tensor(1.0))
            self.pos_spiral_phase = nn.Parameter(torch.tensor(0.0))
            self.pos_spiral_slope = nn.Parameter(torch.tensor(1.0/9.0))
            self.pos_spiral_offset = nn.Parameter(torch.tensor(0.0))
            self.z10_enc = nn.Parameter(torch.zeros(1, pos_dim))
            self.special_enc = nn.Parameter(torch.zeros(3, pos_dim))
            # Initialize from full PE
            full_pe = build_pos_enc_full_3d()[:, :pos_dim]
            self.z10_enc.data[0] = full_pe[Z_START + 10]
            self.special_enc.data[0] = full_pe[PLUS_POS]
            self.special_enc.data[1] = full_pe[EQ_POS]
            self.special_enc.data[2] = full_pe[EOS_POS]
        elif shared_pos == 'xy':
            # X[i] = Y[i] = digit_enc[i], Z has separate encoding
            self.digit_enc = nn.Parameter(torch.zeros(10, pos_dim))  # for X[0..9] and Y[0..9]
            self.z_enc = nn.Parameter(torch.zeros(11, pos_dim))      # for Z[0..10]
            self.special_enc = nn.Parameter(torch.zeros(3, pos_dim)) # +, =, EOS
            self._init_shared_xy()
        elif shared_pos == 'xyz':
            # X[i] = Y[i] = Z[i] = digit_enc[i]
            self.digit_enc = nn.Parameter(torch.zeros(10, pos_dim))  # shared for all
            self.z10_enc = nn.Parameter(torch.zeros(1, pos_dim))     # Z[10] carry overflow
            self.special_enc = nn.Parameter(torch.zeros(3, pos_dim)) # +, =, EOS
            self._init_shared_xyz()
        else:
            self.pos_enc = nn.Parameter(build_pos_enc_full_3d()[:, :pos_dim])

        # Build index mapping for shared modes
        if shared_pos in ('xy', 'xyz'):
            self._build_pos_indices()

        # Attention
        self.tied_qk = tied_qk
        self.q_proj = nn.Linear(pos_dim, self.attn_dim, bias=False)
        if not tied_qk:
            self.k_proj = nn.Linear(pos_dim, self.attn_dim, bias=False)
        self.v_proj = nn.Linear(tok_dim, self.attn_dim, bias=False)
        if tied_vo:
            # out_proj = v_proj.weight.T — maps attn_dim→tok_dim, no extra params
            self.out_proj_dim = tok_dim
        else:
            self.out_proj_dim = out_proj_dim or self.d_model
            self.out_proj = nn.Linear(self.attn_dim, self.out_proj_dim, bias=False)

        # Norms
        self.shared_ln = shared_ln
        NormClass = RMSNorm if norm_type == 'rms' else nn.LayerNorm
        self.ln1 = NormClass(self.d_model)
        if not shared_ln:
            self.ln2 = NormClass(self.d_model)
        if not shared_ln_f:
            self.ln_f = NormClass(self.d_model)
        # shared_ln_f: reuse ln1 as ln_f too

        # FFN
        self.tied_ffn = tied_ffn
        if tied_ffn:
            self.ffn_w = nn.Linear(self.d_model, ffn_dim, bias=ffn_bias)
            self.ffn_bias2 = nn.Parameter(torch.zeros(self.d_model)) if ffn_bias else None
        else:
            self.ffn = nn.Sequential(
                nn.Linear(self.d_model, ffn_dim, bias=ffn_bias),
                nn.GELU(),
                nn.Linear(ffn_dim, self.d_model, bias=ffn_bias),
            )

        # Tied output head
        self.head_proj = nn.Linear(self.d_model, tok_dim, bias=False)

        self.register_buffer("causal_mask",
            torch.tril(torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN)).unsqueeze(0).unsqueeze(0))

        self._init_weights()

    def _init_shared_xy(self):
        """Initialize from full PE, averaging X and Y positions."""
        full_pe = build_pos_enc_full_3d()[:, :self.pos_dim]
        for i in range(10):
            # Average X[i] and Y[i] for digit_enc
            self.digit_enc.data[i] = (full_pe[X_START + i] + full_pe[Y_START + i]) / 2.0
        for i in range(11):
            self.z_enc.data[i] = full_pe[Z_START + i]
        self.special_enc.data[0] = full_pe[PLUS_POS]
        self.special_enc.data[1] = full_pe[EQ_POS]
        self.special_enc.data[2] = full_pe[EOS_POS]

    def _init_shared_xyz(self):
        """Initialize from full PE, averaging X, Y, Z positions."""
        full_pe = build_pos_enc_full_3d()[:, :self.pos_dim]
        for i in range(10):
            self.digit_enc.data[i] = (full_pe[X_START + i] + full_pe[Y_START + i] + full_pe[Z_START + i]) / 3.0
        self.z10_enc.data[0] = full_pe[Z_START + 10]
        self.special_enc.data[0] = full_pe[PLUS_POS]
        self.special_enc.data[1] = full_pe[EQ_POS]
        self.special_enc.data[2] = full_pe[EOS_POS]

    def _build_pos_indices(self):
        """Build index mapping from sequence position to sub-encodings."""
        # For each position: (source, index)
        # source: 0=digit_enc, 1=z_enc/z10, 2=special
        sources = []
        indices = []
        for p in range(MAX_SEQ_LEN):
            if p < 10:  # X digits
                sources.append(0); indices.append(p)
            elif p == 10:  # +
                sources.append(2); indices.append(0)
            elif p < 21:  # Y digits
                if self.shared_pos == 'xy':
                    sources.append(0); indices.append(p - 11)
                else:  # xyz
                    sources.append(0); indices.append(p - 11)
            elif p == 21:  # =
                sources.append(2); indices.append(1)
            elif p < 33:  # Z digits
                z_idx = p - 22
                if self.shared_pos == 'xy':
                    sources.append(1); indices.append(z_idx)
                else:  # xyz
                    if z_idx < 10:
                        sources.append(0); indices.append(z_idx)
                    else:  # z_idx == 10
                        sources.append(1); indices.append(0)
            else:  # EOS (p=33)
                sources.append(2); indices.append(2)
        self.register_buffer('_pos_sources', torch.tensor(sources))
        self.register_buffer('_pos_indices', torch.tensor(indices))

    def _get_digit_pos_enc(self):
        """Get digit position encodings (10, pos_dim) — parametric or free."""
        if self.param_spiral_pos:
            i = torch.arange(10, dtype=torch.float, device=self.pos_spiral_amp.device)
            enc = torch.zeros(10, 3, device=self.pos_spiral_amp.device)
            angle = 2 * math.pi * i / 10 + self.pos_spiral_phase
            enc[:, 0] = self.pos_spiral_amp * torch.cos(angle)
            enc[:, 1] = self.pos_spiral_amp * torch.sin(angle)
            enc[:, 2] = self.pos_spiral_slope * i + self.pos_spiral_offset
            return enc
        return self.digit_enc

    def _get_pos(self, T):
        if self.shared_pos == 'none':
            return self.pos_enc[:T]

        digit_enc = self._get_digit_pos_enc()
        dev = digit_enc.device
        pos = torch.zeros(T, self.pos_dim, device=dev)
        for t in range(T):
            src = self._pos_sources[t].item()
            idx = self._pos_indices[t].item()
            if src == 0:
                pos[t] = digit_enc[idx]
            elif src == 1:
                if self.shared_pos == 'xy':
                    pos[t] = self.z_enc[idx]
                else:  # xyz
                    pos[t] = self.z10_enc[idx]
            else:
                pos[t] = self.special_enc[idx]
        return pos

    def _get_tok_emb_matrix(self):
        """Get the full (14, tok_dim) embedding matrix, parametric or free."""
        if self.param_spiral:
            i = torch.arange(10, dtype=torch.float, device=self.spiral_amp.device)
            emb = torch.zeros(14, 3, device=self.spiral_amp.device)
            angle = 2 * math.pi * i / 10 + self.spiral_phase
            emb[:10, 0] = self.spiral_amp * torch.cos(angle)
            emb[:10, 1] = self.spiral_amp * torch.sin(angle)
            emb[:10, 2] = self.spiral_slope * i + self.spiral_offset
            emb[10:14] = self.special_tok_emb
            return emb
        return self.tok_emb.weight

    def _init_weights(self):
        skip = {'tok_emb.weight', 'pos_enc', 'digit_enc', 'z_enc', 'z10_enc', 'special_enc',
                'spiral_amp', 'spiral_phase', 'spiral_slope', 'spiral_offset', 'special_tok_emb',
                'pos_spiral_amp', 'pos_spiral_phase', 'pos_spiral_slope', 'pos_spiral_offset'}
        for name, p in self.named_parameters():
            if name in skip:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb_mat = self._get_tok_emb_matrix()
        tok = tok_emb_mat[idx]  # index into the embedding matrix
        pos = self._get_pos(T).unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([tok, pos], dim=-1)

        h = self.ln1(x)
        x_pos = h[:, :, self.tok_dim:]
        x_tok = h[:, :, :self.tok_dim]
        q = self.q_proj(x_pos).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k_proj = self.q_proj if self.tied_qk else self.k_proj
        k = k_proj(x_pos).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_tok).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, self.attn_dim)
        if self.tied_vo:
            attn_out = F.linear(out, self.v_proj.weight.T)  # v_proj.weight.T is (tok_dim, attn_dim), maps attn_dim→tok_dim
        else:
            attn_out = self.out_proj(out)
        if self.out_proj_dim < self.d_model:
            # Only update first out_proj_dim dimensions (e.g., tok dims)
            x = x.clone()
            x[:, :, :self.out_proj_dim] = x[:, :, :self.out_proj_dim] + attn_out
        else:
            x = x + attn_out

        h2 = self.ln1(x) if self.shared_ln else self.ln2(x)
        if self.tied_ffn:
            h2 = F.gelu(self.ffn_w(h2))
            h2 = F.linear(h2, self.ffn_w.weight.T, self.ffn_bias2)
        else:
            h2 = self.ffn(h2)
        x = x + h2
        ln_f = self.ln1 if self.shared_ln_f else self.ln_f
        x = ln_f(x)
        return self.head_proj(x) @ self._get_tok_emb_matrix().T

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
    # d5 + shared XY pos (saves 30 from pos_enc)
    'd5_f3_sxy': dict(tok_dim=2, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=3, shared_pos='xy'),
    'd5_f4_sxy': dict(tok_dim=2, pos_dim=3, ffn_dim=4, n_heads=2, head_dim=3, shared_pos='xy'),
    'd5_f2_sxy': dict(tok_dim=2, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xy'),

    # d5 + shared XYZ pos (saves 60 from pos_enc)
    'd5_f3_sxyz': dict(tok_dim=2, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=3, shared_pos='xyz'),
    'd5_f4_sxyz': dict(tok_dim=2, pos_dim=3, ffn_dim=4, n_heads=2, head_dim=3, shared_pos='xyz'),

    # d6 + shared XY pos (our architecture with shared pos)
    'd6_f3_sxy': dict(tok_dim=3, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=3, shared_pos='xy'),
    'd6_f3_sxyz': dict(tok_dim=3, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=3, shared_pos='xyz'),

    # d5 + shared XY + head_dim=2
    'd5_f3_sxy_hd2': dict(tok_dim=2, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=2, shared_pos='xy'),

    # d6 + shared XYZ push variants (smaller FFN / head_dim)
    'd6_f2_sxyz': dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz'),  # ~260p
    'd6_f1_sxyz': dict(tok_dim=3, pos_dim=3, ffn_dim=1, n_heads=2, head_dim=3, shared_pos='xyz'),  # ~247p
    'd6_f3_sxyz_hd2': dict(tok_dim=3, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=2, shared_pos='xyz'),  # ~243p
    'd6_f2_sxyz_hd2': dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=2, shared_pos='xyz'),  # ~230p

    # d6 + shared XYZ + seed variations (to test stochasticity)
    'd6_f3_sxyz_s2': dict(tok_dim=3, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=3, shared_pos='xyz'),
    'd6_f3_sxyz_s3': dict(tok_dim=3, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=3, shared_pos='xyz'),

    # pos_dim=2 experiments: d_model=5 with tok_dim=3 (circle pos + spiral tok)
    # Saves ~48 params vs d6_f3_sxyz by reducing pos from 3D to 2D
    't3p2_f3_sxyz': dict(tok_dim=3, pos_dim=2, ffn_dim=3, n_heads=2, head_dim=3, shared_pos='xyz'),  # ~225p
    't3p2_f4_sxyz': dict(tok_dim=3, pos_dim=2, ffn_dim=4, n_heads=2, head_dim=3, shared_pos='xyz'),  # ~235p
    't3p2_f2_sxyz': dict(tok_dim=3, pos_dim=2, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz'),  # ~215p
    't3p2_f3_sxy': dict(tok_dim=3, pos_dim=2, ffn_dim=3, n_heads=2, head_dim=3, shared_pos='xy'),    # ~251p

    # No FFN bias variants (saves ffn_dim + d_model = 8 params for d6_f2)
    'd6_f2_sxyz_nfb': dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', ffn_bias=False),  # 252p
    'd6_f3_sxyz_nfb': dict(tok_dim=3, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=3, shared_pos='xyz', ffn_bias=False),  # 264p

    # Tied FFN (W2 = W1^T) — saves d_model*ffn_dim params
    'd6_f2_sxyz_tffn': dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', tied_ffn=True),  # ~248p
    'd6_f3_sxyz_tffn': dict(tok_dim=3, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=3, shared_pos='xyz', tied_ffn=True),  # ~255p
    'd6_f2_sxyz_tffn_nfb': dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', tied_ffn=True, ffn_bias=False),  # ~240p

    # Tied Q/K (Q_proj = K_proj) — saves pos_dim*attn_dim = 18 params
    # Works because with shared XYZ pos, Z[i] has same encoding as X[i]/Y[i],
    # so symmetric attention (Q=K) naturally attends to matching positions
    'd6_f2_sxyz_tqk': dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', tied_qk=True),  # ~242p
    'd6_f3_sxyz_tqk': dict(tok_dim=3, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=3, shared_pos='xyz', tied_qk=True),  # ~255p

    # Tied Q/K + tied FFN (maximum sharing)
    'd6_f2_sxyz_tqk_tffn': dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', tied_qk=True, tied_ffn=True),  # ~230p
    'd6_f2_sxyz_tqk_nfb': dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', tied_qk=True, ffn_bias=False),  # ~234p
    'd6_f2_sxyz_tqk_tffn_nfb': dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', tied_qk=True, tied_ffn=True, ffn_bias=False),  # ~222p

    # Shared LN (ln1 = ln2) — saves 12 params
    'd6_f2_sxyz_sln': dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', shared_ln=True),  # ~248p
    'd6_f2_sxyz_tqk_sln': dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', tied_qk=True, shared_ln=True),  # ~230p
    'd6_f2_sxyz_tqk_tffn_sln': dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', tied_qk=True, tied_ffn=True, shared_ln=True),  # ~218p
    'd6_f2_sxyz_tqk_tffn_nfb_sln': dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', tied_qk=True, tied_ffn=True, ffn_bias=False, shared_ln=True),  # ~210p

    # Restricted out_proj (out_proj to tok_dim only) — saves 18 params for d6
    'd6_f2_sxyz_rop': dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', out_proj_dim=3),  # ~242p
    'd6_f2_sxyz_tqk_rop': dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', tied_qk=True, out_proj_dim=3),  # ~224p
    'd6_f3_sxyz_nfb_rop': dict(tok_dim=3, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=3, shared_pos='xyz', ffn_bias=False, out_proj_dim=3),  # ~246p
    'd6_f3_sxyz_nfb_tqk_rop': dict(tok_dim=3, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=3, shared_pos='xyz', ffn_bias=False, tied_qk=True, out_proj_dim=3),  # ~228p

    # Parametric spiral tok_emb (saves 26 params: 42→16)
    'd6_f2_sxyz_ps': dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', param_spiral=True),  # 234p
    'd6_f2_sxyz_ps_tqk': dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', param_spiral=True, tied_qk=True),  # 216p
    'd6_f2_sxyz_ps_tqk_rop': dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', param_spiral=True, tied_qk=True, out_proj_dim=3),  # 198p
    'd6_f3_sxyz_nfb_ps': dict(tok_dim=3, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=3, shared_pos='xyz', ffn_bias=False, param_spiral=True),  # 238p
    'd6_f3_sxyz_nfb_ps_tqk': dict(tok_dim=3, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=3, shared_pos='xyz', ffn_bias=False, param_spiral=True, tied_qk=True),  # 220p

    # === NEW: RMSNorm (saves 6 per norm → 18 total with 3 norms) ===
    'd6_f2_sxyz_rms': dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', norm_type='rms'),
    'd6_f2_sxyz_rms_tqk': dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', norm_type='rms', tied_qk=True),

    # === NEW: Tied V/Out (saves 18-36 from out_proj, implies rop) ===
    'd6_f2_sxyz_tvo': dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', tied_vo=True),
    'd6_f2_sxyz_tvo_tqk': dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', tied_vo=True, tied_qk=True),

    # === NEW: Parametric spiral positions (saves 26 from pos_enc: 42→16) ===
    'd6_f2_sxyz_psp': dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', param_spiral_pos=True),
    'd6_f2_sxyz_psp_tqk': dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', param_spiral_pos=True, tied_qk=True),

    # === RMSNorm + psp (no WD) — 216p ===
    'd6_f2_sxyz_rms_psp': dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', norm_type='rms', param_spiral_pos=True),
    # === RMSNorm only (no WD) — 242p, tests if RMSNorm works without WD ===
    'd6_f2_sxyz_rms': dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', norm_type='rms'),

    # === NEW: All-shared LN (ln1=ln2=ln_f, saves 24 from norms) ===
    'd6_f2_sxyz_asln': dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', shared_ln=True, shared_ln_f=True),
    'd6_f2_sxyz_asln_tqk': dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', shared_ln=True, shared_ln_f=True, tied_qk=True),

    # === NEW: Double spiral (tok+pos parametric spirals) ===
    'd6_f2_sxyz_ps_psp': dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', param_spiral=True, param_spiral_pos=True),
    'd6_f2_sxyz_ps_psp_tqk': dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', param_spiral=True, param_spiral_pos=True, tied_qk=True),

    # === NEW: Kitchen sink combos ===
    # tvo + tqk + ps = maximum compression
    'd6_f2_sxyz_ps_tqk_tvo': dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', param_spiral=True, tied_qk=True, tied_vo=True),
    # tvo + tqk + psp (spiral pos instead of spiral tok)
    'd6_f2_sxyz_psp_tqk_tvo': dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', param_spiral_pos=True, tied_qk=True, tied_vo=True),
    # Double spiral + tqk + tvo
    'd6_f2_sxyz_ps_psp_tqk_tvo': dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', param_spiral=True, param_spiral_pos=True, tied_qk=True, tied_vo=True),
}


def train(variant_name, max_epochs=1000, resume_from=None):
    cfg = VARIANTS[variant_name]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Variant: {variant_name}")
    print(f"Config: {cfg}")

    model = D5SharedPosModel(**cfg).to(device)
    n_trainable, n_total = model.count_parameters()
    print(f"Parameters: {n_trainable} trainable / {n_total} total")
    for name, p in model.named_parameters():
        print(f"  {name}: {list(p.shape)} = {p.numel()}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    if resume_from and os.path.exists(resume_from):
        model.load_state_dict(torch.load(resume_from, weights_only=True))
        print(f"Resumed from {resume_from}")

    val_dataset = AdditionDataset(VAL_SIZE, seed=42)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    best_exact_acc = -1.0
    best_epoch = 0
    patience_counter = 0
    patience = 500
    save_name = f'best_spos_{variant_name}.pt'

    for epoch in range(1, max_epochs + 1):
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

    with open(f'results_spos_{variant_name}.json', 'w') as f:
        json.dump({'variant': variant_name, 'params': n_total, 'best_ea': best_exact_acc,
                   'best_epoch': best_epoch}, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, required=True, choices=list(VARIANTS.keys()))
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint path')
    args = parser.parse_args()
    train(args.variant, max_epochs=args.epochs, resume_from=args.resume)


if __name__ == '__main__':
    main()
