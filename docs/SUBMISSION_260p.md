# AdderBoard Submission: 260-parameter Spiral Addition Transformer

## Author
Jack Cai

## Unique Parameter Count
260

## Accuracy
**100.00%** (10010/10010 on official AdderBoard test: 10 edge cases + 10,000 random pairs, seed=2025)

Cross-validation: 99.97% (seed=999, 10K random pairs)

## Verification Output
```
Model: Spiral Addition Transformer (shared-XYZ, tied-proj, FFN2+WD)
Author: Jack Cai
Parameters (unique): 260
Architecture: 1L decoder, d=6 (3 tok + 3 pos), 2h, hd=3, ff=2, shared XYZ pos, tied output

Results: 10010/10010 correct (100.00%)
Time: 63.5s (158 additions/sec)
Status: QUALIFIED (threshold: 99%)
```

## Method

1-layer decoder-only transformer with a split-dimension architecture. The model uses `d_model=6` split into two 3D subspaces: one for token values and one for positional information. These are concatenated (not summed) and flow through the model together.

### Architecture
- **d_model=6** = tok_dim(3) + pos_dim(3), concatenated
- **2 attention heads**, head_dim=3
- **FFN inner dim=2** (minimal, enabled by weight decay)
- **Split-head attention**: Q,K project from positional dims; V projects from token dims
- **Tied output head**: `Linear(6,3) @ tok_emb.T` instead of `Linear(6,14)`
- **Shared XYZ positions**: X[i]=Y[i]=Z[i]=digit_enc[i]
- **LayerNorm** (3 independent norms)
- **LSB-first** digit ordering

### Key Tricks
1. **Spiral-initialized embeddings**: Token embeddings initialized as `(cos(2*pi*d/10), sin(2*pi*d/10), d/9)` for digits 0-9, providing geometric structure
2. **Split-head attention**: Q,K attend by position, V reads token values — natural factorization for addition
3. **Shared XYZ positional encoding**: Same learned position vectors for all three operand positions (X, Y, Z), saving 60 parameters
4. **Tied output head**: Project to 3D token subspace then dot with token embedding table, saving 66 parameters
5. **FFN dim=2 + weight decay 0.01**: Extreme FFN compression enabled by AdamW weight decay for grokking
6. **LSB-first ordering**: Digits fed least-significant-first for natural carry propagation

### Training
- Optimizer: AdamW (lr=1e-3, weight_decay=0.01)
- Batch size: 512
- Training samples: 50K per epoch
- Convergence: ~500 epochs via grokking
- Hardware: Single CPU (model is 260 parameters)

## Link to Code
https://github.com/jackcai0/smallest_addition_TF

- Submission: `submission_260.py`
- Checkpoint: `best_model_260.pt`
- Training: `experiments/scripts/train_spos_wd.py --variant d6_f2_sxyz_wd01`

## Additional Notes
- This model achieves **100.00% accuracy** on all 10,010 test cases — zero failures
- The previous trained-weights leaderboard leader had 311 parameters at 99.999%
- Code written with assistance from Claude Code
