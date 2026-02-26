"""Shared position encoding + weight decay experiments.

Combining two key insights:
1. X/Y positions can be shared (they barely diverge in training)
2. Weight decay (0.01) is essential for escaping 10% EA trap in small models

Target param counts:
  d6_f3_sxy_wd01: 303p (proven arch + shared XY + WD)
  d5_f3_sxy_wd01: 256p (tok_dim=2 + shared XY + WD)
  d5_f4_sxy_wd01: 267p (tok_dim=2 + shared XY + FFN4 + WD)
  d5_f2_sxy_wd01: 245p (tok_dim=2 + shared XY + FFN2 + WD)
"""

import argparse, math, os, json
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict
from data import AdditionDataset
from config import (
    VOCAB_SIZE, MAX_SEQ_LEN, MAX_DIGITS, ANSWER_LEN, EQ_POS,
    BATCH_SIZE, VAL_SIZE, TRAIN_SAMPLES_PER_EPOCH,
)
from train_d5_spos import D5SharedPosModel
from train_d5 import evaluate


VARIANTS = {
    # Most promising: proven d6 arch + shared XY pos + WD
    'd6_f3_sxy_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=3, shared_pos='xy'),
        lr=1e-3, wd=0.01),

    # d5 + shared XY + WD (different FFN dims)
    'd5_f4_sxy_wd01': dict(
        model=dict(tok_dim=2, pos_dim=3, ffn_dim=4, n_heads=2, head_dim=3, shared_pos='xy'),
        lr=1e-3, wd=0.01),
    'd5_f3_sxy_wd01': dict(
        model=dict(tok_dim=2, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=3, shared_pos='xy'),
        lr=1e-3, wd=0.01),
    'd5_f2_sxy_wd01': dict(
        model=dict(tok_dim=2, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xy'),
        lr=1e-3, wd=0.01),

    # Also try full pos (no sharing) + WD for reference
    'd5_f2_wd01': dict(
        model=dict(tok_dim=2, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='none'),
        lr=1e-3, wd=0.01),

    # Shared XYZ + WD (likely won't work based on analysis, but try)
    'd6_f3_sxyz_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=3, shared_pos='xyz'),
        lr=1e-3, wd=0.01),
    'd5_f3_sxyz_wd01': dict(
        model=dict(tok_dim=2, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=3, shared_pos='xyz'),
        lr=1e-3, wd=0.01),

    # d5 sxy + hd2 + WD (smallest possible)
    'd5_f3_sxy_hd2_wd': dict(
        model=dict(tok_dim=2, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=2, shared_pos='xy'),
        lr=1e-3, wd=0.01),

    # d6 sxyz + WD push variants
    'd6_f2_sxyz_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz'),
        lr=1e-3, wd=0.01),
    'd6_f1_sxyz_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=1, n_heads=2, head_dim=3, shared_pos='xyz'),
        lr=1e-3, wd=0.01),

    # d5 sxyz + WD (most aggressive)
    'd5_f3_sxyz_wd01': dict(
        model=dict(tok_dim=2, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=3, shared_pos='xyz'),
        lr=1e-3, wd=0.01),
    'd5_f4_sxyz_wd01': dict(
        model=dict(tok_dim=2, pos_dim=3, ffn_dim=4, n_heads=2, head_dim=3, shared_pos='xyz'),
        lr=1e-3, wd=0.01),

    # pos_dim=2 with WD experiments
    't3p2_f3_sxyz_wd01': dict(
        model=dict(tok_dim=3, pos_dim=2, ffn_dim=3, n_heads=2, head_dim=3, shared_pos='xyz'),
        lr=1e-3, wd=0.01),
    't3p2_f2_sxyz_wd01': dict(
        model=dict(tok_dim=3, pos_dim=2, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz'),
        lr=1e-3, wd=0.01),

    # No FFN bias + WD (key: does removing FFN bias break WD-assisted convergence?)
    'd6_f2_sxyz_nfb_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', ffn_bias=False),
        lr=1e-3, wd=0.01),  # 252p
    'd6_f3_sxyz_nfb_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=3, shared_pos='xyz', ffn_bias=False),
        lr=1e-3, wd=0.01),  # 264p
    'd6_f3_sxyz_nfb_tqk_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=3, shared_pos='xyz', ffn_bias=False, tied_qk=True),
        lr=1e-3, wd=0.01),  # 246p
    'd6_f3_sxyz_nfb_tqk_sln_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=3, shared_pos='xyz', ffn_bias=False, tied_qk=True, shared_ln=True),
        lr=1e-3, wd=0.01),  # 234p
    'd6_f3_sxyz_nfb_tqk_tffn_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=3, shared_pos='xyz', ffn_bias=False, tied_qk=True, tied_ffn=True),
        lr=1e-3, wd=0.01),  # 228p
    'd6_f3_sxyz_nfb_tqk_tffn_sln_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=3, shared_pos='xyz', ffn_bias=False, tied_qk=True, tied_ffn=True, shared_ln=True),
        lr=1e-3, wd=0.01),  # 216p
    'd6_f3_sxyz_nfb_tffn_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=3, shared_pos='xyz', ffn_bias=False, tied_ffn=True),
        lr=1e-3, wd=0.01),  # 246p (same as tqk variant)

    # pos_dim=2 + WD + no FFN bias (most aggressive)
    't3p2_f2_sxyz_nfb_wd01': dict(
        model=dict(tok_dim=3, pos_dim=2, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', ffn_bias=False),
        lr=1e-3, wd=0.01),  # ~207p

    # head_dim=2 + WD (since WD enabled FFN=2, maybe also enables hd=2?)
    'd6_f3_sxyz_hd2_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=2, shared_pos='xyz'),
        lr=1e-3, wd=0.01),  # ~243p
    'd6_f2_sxyz_hd2_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=2, shared_pos='xyz'),
        lr=1e-3, wd=0.01),  # ~230p

    # Tied FFN + WD
    'd6_f2_sxyz_tffn_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', tied_ffn=True),
        lr=1e-3, wd=0.01),  # ~248p
    'd6_f3_sxyz_tffn_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=3, shared_pos='xyz', tied_ffn=True),
        lr=1e-3, wd=0.01),  # ~255p
    'd6_f2_sxyz_tffn_nfb_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', tied_ffn=True, ffn_bias=False),
        lr=1e-3, wd=0.01),  # ~240p

    # Tied Q/K + WD (Q_proj = K_proj, saves 18 params)
    'd6_f2_sxyz_tqk_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', tied_qk=True),
        lr=1e-3, wd=0.01),  # ~242p
    'd6_f3_sxyz_tqk_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=3, shared_pos='xyz', tied_qk=True),
        lr=1e-3, wd=0.01),  # ~255p

    # Tied Q/K + tied FFN + WD (maximum sharing)
    'd6_f2_sxyz_tqk_tffn_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', tied_qk=True, tied_ffn=True),
        lr=1e-3, wd=0.01),  # ~230p
    'd6_f2_sxyz_tqk_nfb_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', tied_qk=True, ffn_bias=False),
        lr=1e-3, wd=0.01),  # ~234p
    'd6_f2_sxyz_tqk_tffn_nfb_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', tied_qk=True, tied_ffn=True, ffn_bias=False),
        lr=1e-3, wd=0.01),  # ~222p

    # Shared LN + WD (ln1 = ln2, saves 12 params)
    'd6_f2_sxyz_sln_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', shared_ln=True),
        lr=1e-3, wd=0.01),  # ~248p
    'd6_f2_sxyz_tqk_sln_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', tied_qk=True, shared_ln=True),
        lr=1e-3, wd=0.01),  # ~230p
    'd6_f2_sxyz_tqk_tffn_sln_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', tied_qk=True, tied_ffn=True, shared_ln=True),
        lr=1e-3, wd=0.01),  # ~218p
    'd6_f2_sxyz_tqk_tffn_nfb_sln_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', tied_qk=True, tied_ffn=True, ffn_bias=False, shared_ln=True),
        lr=1e-3, wd=0.01),  # ~210p

    # Restricted out_proj (to tok_dim only) + WD — saves 18 params
    'd6_f2_sxyz_rop_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', out_proj_dim=3),
        lr=1e-3, wd=0.01),  # ~242p
    'd6_f2_sxyz_tqk_rop_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', tied_qk=True, out_proj_dim=3),
        lr=1e-3, wd=0.01),  # ~224p
    'd6_f3_sxyz_nfb_rop_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=3, shared_pos='xyz', ffn_bias=False, out_proj_dim=3),
        lr=1e-3, wd=0.01),  # ~246p
    'd6_f3_sxyz_nfb_tqk_rop_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=3, shared_pos='xyz', ffn_bias=False, tied_qk=True, out_proj_dim=3),
        lr=1e-3, wd=0.01),  # ~228p

    # Parametric spiral + WD (saves 26 params from tok_emb)
    'd6_f2_sxyz_ps_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', param_spiral=True),
        lr=1e-3, wd=0.01),  # 234p
    'd6_f2_sxyz_ps_tqk_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', param_spiral=True, tied_qk=True),
        lr=1e-3, wd=0.01),  # 216p
    'd6_f2_sxyz_ps_tqk_rop_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', param_spiral=True, tied_qk=True, out_proj_dim=3),
        lr=1e-3, wd=0.01),  # 198p
    'd6_f3_sxyz_nfb_ps_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=3, shared_pos='xyz', ffn_bias=False, param_spiral=True),
        lr=1e-3, wd=0.01),  # 238p
    'd6_f3_sxyz_nfb_ps_tqk_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=3, shared_pos='xyz', ffn_bias=False, param_spiral=True, tied_qk=True),
        lr=1e-3, wd=0.01),  # 220p

    # === NEW ROUND: RMSNorm, tied V/Out, spiral pos, all-shared LN ===

    # RMSNorm + WD (saves 18p from norms)
    'd6_f2_sxyz_rms_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', norm_type='rms'),
        lr=1e-3, wd=0.01),  # 242p
    'd6_f2_sxyz_rms_tqk_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', norm_type='rms', tied_qk=True),
        lr=1e-3, wd=0.01),  # 224p

    # Tied V/Out + WD (saves 36p from out_proj entirely)
    'd6_f2_sxyz_tvo_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', tied_vo=True),
        lr=1e-3, wd=0.01),  # 224p
    'd6_f2_sxyz_tvo_tqk_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', tied_vo=True, tied_qk=True),
        lr=1e-3, wd=0.01),  # 206p

    # Parametric spiral positions + WD (saves 26p from pos_enc)
    'd6_f2_sxyz_psp_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', param_spiral_pos=True),
        lr=1e-3, wd=0.01),  # 234p
    'd6_f2_sxyz_psp_tqk_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', param_spiral_pos=True, tied_qk=True),
        lr=1e-3, wd=0.01),  # 216p

    # All-shared LN + WD (ln1=ln2=ln_f, saves 24p)
    'd6_f2_sxyz_asln_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', shared_ln=True, shared_ln_f=True),
        lr=1e-3, wd=0.01),  # 236p
    'd6_f2_sxyz_asln_tqk_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', shared_ln=True, shared_ln_f=True, tied_qk=True),
        lr=1e-3, wd=0.01),  # 218p

    # Double spiral (tok+pos) + WD
    'd6_f2_sxyz_ps_psp_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', param_spiral=True, param_spiral_pos=True),
        lr=1e-3, wd=0.01),  # 208p
    'd6_f2_sxyz_ps_psp_tqk_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', param_spiral=True, param_spiral_pos=True, tied_qk=True),
        lr=1e-3, wd=0.01),  # 190p

    # Aggressive combos
    'd6_f2_sxyz_ps_tqk_tvo_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', param_spiral=True, tied_qk=True, tied_vo=True),
        lr=1e-3, wd=0.01),  # 180p
    'd6_f2_sxyz_psp_tqk_tvo_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', param_spiral_pos=True, tied_qk=True, tied_vo=True),
        lr=1e-3, wd=0.01),  # 180p
    'd6_f2_sxyz_ps_psp_tqk_tvo_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', param_spiral=True, param_spiral_pos=True, tied_qk=True, tied_vo=True),
        lr=1e-3, wd=0.01),  # 154p

    # RMSNorm combos (without tvo — tvo confirmed failure)
    'd6_f2_sxyz_rms_tvo_tqk_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', norm_type='rms', tied_vo=True, tied_qk=True),
        lr=1e-3, wd=0.01),  # 188p
    'd6_f2_sxyz_rms_ps_tqk_tvo_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', norm_type='rms', param_spiral=True, tied_qk=True, tied_vo=True),
        lr=1e-3, wd=0.01),  # 162p

    # === ROUND 3: Best combos without tvo ===

    # RMSNorm + spiral pos + tqk (most promising combo)
    'd6_f2_sxyz_rms_psp_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', norm_type='rms', param_spiral_pos=True),
        lr=1e-3, wd=0.01),  # 216p
    'd6_f2_sxyz_rms_psp_tqk_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', norm_type='rms', param_spiral_pos=True, tied_qk=True),
        lr=1e-3, wd=0.01),  # 198p

    # RMSNorm + parametric spiral tok + tqk
    'd6_f2_sxyz_rms_ps_tqk_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', norm_type='rms', param_spiral=True, tied_qk=True),
        lr=1e-3, wd=0.01),  # 198p
    'd6_f2_sxyz_rms_ps_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', norm_type='rms', param_spiral=True),
        lr=1e-3, wd=0.01),  # 216p

    # Spiral pos + shared LN (both show promise)
    'd6_f2_sxyz_psp_sln_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', param_spiral_pos=True, shared_ln=True),
        lr=1e-3, wd=0.01),  # 222p
    'd6_f2_sxyz_psp_sln_tqk_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', param_spiral_pos=True, shared_ln=True, tied_qk=True),
        lr=1e-3, wd=0.01),  # 204p

    # RMSNorm + spiral pos + shared LN → ultimate combo
    'd6_f2_sxyz_rms_psp_sln_tqk_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', norm_type='rms', param_spiral_pos=True, shared_ln=True, tied_qk=True),
        lr=1e-3, wd=0.01),  # 192p?

    # Spiral pos + restricted out_proj (rop works, tvo doesn't)
    'd6_f2_sxyz_psp_rop_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', param_spiral_pos=True, out_proj_dim=3),
        lr=1e-3, wd=0.01),  # 216p
    'd6_f2_sxyz_psp_tqk_rop_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', param_spiral_pos=True, tied_qk=True, out_proj_dim=3),
        lr=1e-3, wd=0.01),  # 198p

    # RMSNorm + shared LN
    'd6_f2_sxyz_rms_sln_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', norm_type='rms', shared_ln=True),
        lr=1e-3, wd=0.01),  # 236p
    'd6_f2_sxyz_rms_sln_tqk_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', norm_type='rms', shared_ln=True, tied_qk=True),
        lr=1e-3, wd=0.01),  # 218p

    # === ROUND 4: FFN=3 path with RMSNorm/spiral (proven: nfb+f3+WD=264p, 99.96%) ===

    # FFN=3 + nfb + rms + ps + WD
    'd6_f3_sxyz_nfb_rms_ps_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=3, shared_pos='xyz', ffn_bias=False, norm_type='rms', param_spiral=True),
        lr=1e-3, wd=0.01),  # 220p
    # FFN=3 + nfb + rms + psp + WD
    'd6_f3_sxyz_nfb_rms_psp_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=3, shared_pos='xyz', ffn_bias=False, norm_type='rms', param_spiral_pos=True),
        lr=1e-3, wd=0.01),  # 220p
    # FFN=3 + nfb + rms + ps + tqk + WD
    'd6_f3_sxyz_nfb_rms_ps_tqk_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=3, shared_pos='xyz', ffn_bias=False, norm_type='rms', param_spiral=True, tied_qk=True),
        lr=1e-3, wd=0.01),  # 202p
    # FFN=3 + nfb + rms only + WD
    'd6_f3_sxyz_nfb_rms_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=3, shared_pos='xyz', ffn_bias=False, norm_type='rms'),
        lr=1e-3, wd=0.01),  # 246p
    # FFN=3 + nfb + ps only + WD (no rms)
    # Already exists as d6_f3_sxyz_nfb_ps_wd01 (238p)

    # FFN=2 + rms + ps + sln + WD
    'd6_f2_sxyz_rms_ps_sln_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', norm_type='rms', param_spiral=True, shared_ln=True),
        lr=1e-3, wd=0.01),  # 210p
    # FFN=2 + rms + ps + asln + WD
    'd6_f2_sxyz_rms_ps_asln_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', norm_type='rms', param_spiral=True, shared_ln=True, shared_ln_f=True),
        lr=1e-3, wd=0.01),  # 204p

    # === ROUND 5: WD and LR variations for 216p (breaking 92% plateau) ===

    # Higher WD for rms+ps
    'd6_f2_sxyz_rms_ps_wd02': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', norm_type='rms', param_spiral=True),
        lr=1e-3, wd=0.02),  # 216p, stronger WD
    'd6_f2_sxyz_rms_ps_wd03': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', norm_type='rms', param_spiral=True),
        lr=1e-3, wd=0.03),  # 216p, even stronger WD
    'd6_f2_sxyz_rms_ps_wd005': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', norm_type='rms', param_spiral=True),
        lr=1e-3, wd=0.005),  # 216p, lighter WD

    # Lower LR for rms+ps
    'd6_f2_sxyz_rms_ps_lr4': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=2, n_heads=2, head_dim=3, shared_pos='xyz', norm_type='rms', param_spiral=True),
        lr=3e-4, wd=0.01),  # 216p, lower LR

    # FFN=3 + nfb + rms + ps with different WD
    'd6_f3_sxyz_nfb_rms_ps_wd02': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=3, shared_pos='xyz', ffn_bias=False, norm_type='rms', param_spiral=True),
        lr=1e-3, wd=0.02),  # 220p
    'd6_f3_sxyz_nfb_rms_ps_wd005': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=3, shared_pos='xyz', ffn_bias=False, norm_type='rms', param_spiral=True),
        lr=1e-3, wd=0.005),  # 220p
}


def train(variant_name, max_epochs=1000, resume_from=None):
    cfg = VARIANTS[variant_name]
    model_cfg = cfg['model']
    lr = cfg['lr']
    wd = cfg['wd']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Variant: {variant_name}")
    print(f"Model: {model_cfg}")
    print(f"LR: {lr}, WD: {wd}")

    model = D5SharedPosModel(**model_cfg).to(device)
    n_trainable, n_total = model.count_parameters()
    print(f"Parameters: {n_trainable} trainable / {n_total} total")
    for name, p in model.named_parameters():
        print(f"  {name}: {list(p.shape)} = {p.numel()}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # Resume from checkpoint if specified
    start_epoch = 1
    if resume_from and os.path.exists(resume_from):
        ckpt = torch.load(resume_from, weights_only=False)
        if isinstance(ckpt, dict) and 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
            if 'optimizer' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
            start_epoch = ckpt.get('epoch', 0) + 1
            print(f"Resumed from {resume_from} at epoch {start_epoch}")
        else:
            model.load_state_dict(ckpt)
            print(f"Resumed model weights from {resume_from}")

    val_dataset = AdditionDataset(VAL_SIZE, seed=42)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    best_exact_acc = -1.0
    best_epoch = 0
    patience_counter = 0
    patience = 500
    save_name = f'best_spwd_{variant_name}.pt'
    ckpt_name = f'ckpt_spwd_{variant_name}.pt'

    for epoch in range(start_epoch, max_epochs + 1):
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

        # Save full checkpoint every 100 epochs for resuming
        if epoch % 100 == 0:
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'epoch': epoch, 'best_ea': best_exact_acc}, ckpt_name)

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

    with open(f'results_spwd_{variant_name}.json', 'w') as f:
        json.dump({'variant': variant_name, 'params': n_total, 'best_ea': best_exact_acc,
                   'best_epoch': best_epoch, 'wd': wd}, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, required=True, choices=list(VARIANTS.keys()))
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint path')
    args = parser.parse_args()
    train(args.variant, max_epochs=args.epochs, resume_from=args.resume)


if __name__ == '__main__':
    main()
