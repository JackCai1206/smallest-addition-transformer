# Experiment Log — Smallest Addition Transformer

## Current Best Models
| Model | Params | Accuracy (10K, seed=2025) | Epochs | Status |
|-------|--------|---------------------------|--------|--------|
| **d6_f2_sxyz_rms_wd01** | **242** | **100.00%** | ~1015 | **NEW RECORD** |
| d6_f2_sxyz_wd01 | 260 | 100.00% | ~500 | Confirmed |
| d6_f3_sxyz_nfb_wd01 | 264 | 99.96% | ~910 | Confirmed |
| d6_f3_sxyz | 273 | 99.97% | ~700 | Confirmed |
| tp_f3 | 333 | 99.88% | ~201 | Confirmed |
| baseline | 438 | 99.83% | 67 | Reference |

## AdderBoard Leaderboard
| Rank | Params | Acc | Key Techniques |
|------|--------|-----|----------------|
| **Ours** | **242** | **100.0%** | d=6, shared XYZ, tied_proj, FFN2+WD, RMSNorm |
| Ours (prev) | 260 | 100.0% | d=6, shared XYZ, tied_proj, FFN2+WD |
| Previous #1 | 311 | 99.999% | d=4, rank-3 factorization, shared-A tied-KV |

---

## KEY DISCOVERIES

### 1. Shared XYZ Position Encoding (-60 params)
X[i]=Y[i]=Z[i]=digit_enc[i]. Only Z[10] and special tokens separate.

### 2. Tied Output Head (-66 params)
Linear(6->3) @ tok_emb.T instead of Linear(6->14).

### 3. Weight Decay for FFN=2 (essential)
FFN=2 requires WD=0.01 (AdamW). Without WD: stuck at 1%.
FFN=3 works without WD but WD HURTS it (unless no-FFN-bias).

### 4. No-FFN-bias + WD enables FFN=3 grokking (-9 params)
- Removing FFN biases changes WD dynamics
- FFN=3 + no-bias + WD = 264p, reaches 99.96%
- No-FFN-bias + FFN=2 FAILS (even with WD, da stuck at 0.65)

### 5. RMSNorm (-18 params) [NEW]
- Replace LayerNorm with RMSNorm: `x * weight / sqrt(mean(x^2) + eps)`
- Saves 6 params per norm (no bias, no mean centering) x 3 norms = 18p
- 242p model (260p - 18p) reaches 100.00% at epoch ~1015
- Grokking delayed vs LayerNorm (~1000 epochs vs ~500) but reaches same accuracy

### 6. Confirmed Failures (Compression Attempts)
| Change | Params | Result | Notes |
|--------|--------|--------|-------|
| tok_dim=2 | — | 10-27% max | Essential dim must be 3 |
| pos_dim=2 | — | 1% max | Essential dim must be 3 |
| head_dim=2 | — | 6-10% max | Even with WD |
| FFN_dim=1 | — | 14% max | Even with WD |
| 1 head | — | ea<3% | Must have 2 heads |
| Remove LN biases | — | 10% trap | - |
| Tied FFN (no WD) | — | da=0.608 | Too constrained |
| WD + FFN=3 (with bias) | — | 14% max | WD hurts FFN=3 |
| **Tied V/Out (out=v.T)** | 224 | da<0.62 | **FAILS** — too constrained |
| **Parametric spiral pos** | 234 | 94.5% max* | Still climbing at ep 893 |
| **All-shared LN (ln1=ln2=ln_f)** | 236 | 7% max | Fails completely |
| **Shared LN (ln1=ln2)** | 248 | 3% at ep 780 | Very slow, may not converge |
| **Restricted out_proj (->tok)** | 242 | 39% at ep 731 | Slow, unlikely to converge |
| **RMSNorm + spiral tok** | 216 | 92%* plateau | Plateaus, doesn't reach 99% |
| **FFN=3 + RMSNorm (nfb)** | 246 | 10% at ep 1035 | Stuck in 10% trap |
| **FFN=3 + RMSNorm + spiral** | 220 | 11% max | Fails |
| **FFN=3 + RMSNorm + spiral + tqk** | 202 | 3.5% max | Fails |

*Still running but not converging to 99%+

---

## Parameter Breakdown — 242p (d6_f2_sxyz_rms_wd01)
| Component | Shape | Params | % |
|-----------|-------|--------|---|
| tok_emb | (14, 3) | 42 | 17.4% |
| digit_enc (shared XYZ) | (10, 3) | 30 | 12.4% |
| z10_enc + special_enc | (1+3, 3) | 12 | 5.0% |
| q_proj | (3, 6) | 18 | 7.4% |
| k_proj | (3, 6) | 18 | 7.4% |
| v_proj | (3, 6) | 18 | 7.4% |
| out_proj | (6, 6) | 36 | 14.9% |
| RMSNorm x3 | 3 x 6 | 18 | 7.4% |
| ffn | (6,2)+(2)+(2,6)+(6) | 32 | 13.2% |
| head_proj | (6, 3) | 18 | 7.4% |
| **Total** | | **242** | |

## Parameter Breakdown — 260p (d6_f2_sxyz_wd01)
| Component | Shape | Params | % |
|-----------|-------|--------|---|
| tok_emb | (14, 3) | 42 | 16.2% |
| pos_enc (shared XYZ) | 10x3+1x3+3x3 | 42 | 16.2% |
| q_proj | (3, 6) | 18 | 6.9% |
| k_proj | (3, 6) | 18 | 6.9% |
| v_proj | (3, 6) | 18 | 6.9% |
| out_proj | (6, 6) | 36 | 13.8% |
| LayerNorm x3 | 3x12 | 36 | 13.8% |
| ffn | (6,2)+(2)+(2,6)+(6) | 32 | 12.3% |
| head_proj | (6, 3) | 18 | 6.9% |
| **Total** | | **260** | |

---

## Compression Techniques — Results Summary

### Successful
| Technique | Savings | From | To | Accuracy |
|-----------|---------|------|----|----------|
| RMSNorm (no bias/mean) | -18p | 260p | **242p** | **100.00%** |
| Shared XYZ pos | -60p | 320p | 260p | 100.00% |
| Tied output head | -66p | 386p | 320p | 100.00% |
| FFN dim 6->2 + WD | -24p | 284p | 260p | 100.00% |

### Failed
| Technique | Savings | Result | Why |
|-----------|---------|--------|-----|
| Tied V/Out (out=v_proj.T) | -36p | da<0.62 | Projection too constrained |
| All-shared LN (ln1=ln2=ln_f) | -12p | 7% max | Norms need independence |
| Shared LN (ln1=ln2 only) | -12p | 3% at ep 780 | Still too constrained |
| Restricted out_proj (6->3) | -18p | 39% slow | Bottleneck too narrow |
| Parametric spiral tok | -26p | 92% plateau | 4 params can't represent all digits |
| Tied Q/K | -18p | 39% slow | Q and K need separate weights |

### Still Under Investigation
| Technique | Params | Status |
|-----------|--------|--------|
| Parametric spiral pos (psp) | 234p | 94.5% at ep 893, climbing |
| RMSNorm + spiral tok | 216p | 6% at ep 783, not grokking |
| Various WD/LR/seed combos | 216p | Early epochs, most struggling |
