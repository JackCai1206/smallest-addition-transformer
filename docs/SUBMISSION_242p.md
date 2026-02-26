# AdderBoard Submission: 242-parameter Spiral Addition Transformer (RMSNorm)

## Author
Jack Cai

## Unique Parameter Count
242

## Accuracy
**99.07%** (9917/10010 on official AdderBoard test: 10 edge cases + 10,000 random pairs, seed=2025)

Cross-validation: 99.80% (seed=999, 10K random pairs)

## Verification Output
```
Model: Spiral Addition Transformer (RMSNorm, shared-XYZ, tied-proj, FFN2+WD)
Author: Jack Cai
Parameters (unique): 242
Architecture: 1L decoder, d=6 (3 tok + 3 pos), 2h, hd=3, ff=2, RMSNorm, shared XYZ pos, tied output

Results: 9917/10010 correct (99.07%)
Time: 71.4s (140 additions/sec)
Status: QUALIFIED (threshold: 99%)
```

## Method

Same architecture as the 260p model but with **RMSNorm replacing LayerNorm**, saving 18 parameters (6 per norm layer x 3 norms). RMSNorm uses `x * weight / sqrt(mean(x^2) + eps)` — no learned bias, no mean centering.

### Architecture
- **d_model=6** = tok_dim(3) + pos_dim(3), concatenated
- **2 attention heads**, head_dim=3
- **FFN inner dim=2** (minimal, enabled by weight decay)
- **Split-head attention**: Q,K project from positional dims; V projects from token dims
- **Tied output head**: `Linear(6,3) @ tok_emb.T` instead of `Linear(6,14)`
- **Shared XYZ positions**: X[i]=Y[i]=Z[i]=digit_enc[i]
- **RMSNorm** (3 independent norms, no bias — saves 18 params vs LayerNorm)
- **LSB-first** digit ordering

### Key Tricks
1. **Spiral-initialized embeddings**: `(cos(2*pi*d/10), sin(2*pi*d/10), d/9)` for digits 0-9
2. **Split-head attention**: Q,K attend by position, V reads token values
3. **Shared XYZ positional encoding**: Same position vectors for X, Y, Z (-60 params)
4. **Tied output head**: Project to 3D token subspace then dot with embeddings (-66 params)
5. **RMSNorm**: No bias, no mean centering — saves 18 params vs LayerNorm
6. **FFN dim=2 + weight decay 0.01**: Extreme compression enabled by grokking
7. **LSB-first ordering**: Natural carry propagation

### Training
- Optimizer: AdamW (lr=1e-3, weight_decay=0.01)
- Batch size: 512
- Training samples: 50K per epoch
- Convergence: ~1000 epochs via grokking (slower than 260p due to RMSNorm)
- Hardware: Single CPU

## Link to Code
https://github.com/jackcai0/smallest_addition_TF

- Submission: `submission_242.py`
- Checkpoint: `best_model_242.pt`
- Training: `experiments/scripts/train_spos_wd.py --variant d6_f2_sxyz_rms_wd01`

## Additional Notes
- 242p = 260p - 18 (RMSNorm replaces LayerNorm)
- Grokking takes ~1000 epochs vs ~500 for the 260p LayerNorm version
- 93 failures are mostly carry-chain errors on 10-digit numbers (11th digit prediction)
- A fresh training run is in progress to potentially improve the checkpoint
- Code written with assistance from Claude Code
