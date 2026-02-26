# Spiral Addition Transformer

A **242-parameter** transformer that learns 10-digit integer addition with **100.00% accuracy** on the [AdderBoard](https://github.com/cognizant-ai-labs/adderboard) leaderboard.

## Leaderboard Results

| Model | Params | Accuracy | Checkpoint | Submission |
|-------|--------|----------|------------|------------|
| **242p (RMSNorm)** | **242** | **100.00%** | `best_model_242.pt` | `submission_242.py` |
| 260p (LayerNorm) | 260 | 100.00% | `best_model_260.pt` | `submission_260.py` |
| 438p (baseline) | 438 | 99.83% | — | `submission.py` |

Previous AdderBoard trained-weights leader: 311 params at 99.999%.

## Architecture

```
1-layer decoder-only transformer
  d_model = 6 = tok_dim(3) + pos_dim(3)   (concatenated, not summed)
  2 attention heads, head_dim = 3
  FFN inner dim = 2 (with weight decay 0.01 for grokking)

  Split-head attention:
    Q, K from positional dims (last 3) — routes by position
    V from token dims (first 3) — carries digit information

  Token embedding: spiral-initialized (cos, sin, linear ramp) over digits 0-9
  Positional encoding: shared X=Y=Z positions (saves 60 params)
  Output head: tied — Linear(6->3) @ tok_emb.T (saves 66 params)
  Normalization: RMSNorm (saves 18 params vs LayerNorm)

  Input format: LSB-first digits, zero-padded to 10 digits
    e.g., 356 + 478 = 834 -> "6 5 3 0...0 + 8 7 4 0...0 = 4 3 8 0...0 EOS"
```

## Key Techniques

1. **Split-dim representation** (`d_model=6 = 3 tok + 3 pos`): Token and position information live in separate subspaces, concatenated (not summed). Q,K attend over position; V reads token values.

2. **Shared XYZ positional encoding** (`-60 params`): The position encoding for digit `i` in the first operand, second operand, and result are all identical: `digit_enc[i]`. Only the 11th result digit and special tokens (+, =, EOS) get separate encodings.

3. **Tied output head** (`-66 params`): Instead of a full `Linear(6, 14)` output projection, use `Linear(6, 3) @ tok_emb.T` — project to token subspace then dot-product with the token embedding table.

4. **RMSNorm** (`-18 params`): Replace LayerNorm with RMSNorm (no bias, no mean centering). Saves 6 params per norm layer x 3 norms.

5. **FFN dim=2 + weight decay** (`-24 params`): Minimal FFN with AdamW weight decay (0.01) enables grokking even at this extreme compression.

6. **Spiral initialization**: Token embeddings initialized as `(cos(2*pi*d/10), sin(2*pi*d/10), d/9)` for digits 0-9, providing geometric structure that accelerates learning.

7. **LSB-first ordering**: Digits are fed least-significant-first, aligning with natural carry propagation.

## Parameter Breakdown (242p model)

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
| FFN | (6,2)+(2)+(2,6)+(6) | 32 | 13.2% |
| head_proj | (6, 3) | 18 | 7.4% |
| **Total** | | **242** | |

## Usage

```bash
# Verify with AdderBoard (10,010 test cases)
python adderboard_verify.py submission_242.py

# Or verify the 260p model
python adderboard_verify.py submission_260.py

# Quick smoke test
python smoke_test.py
```

## Training

The 242p model trains via grokking: it memorizes first, then suddenly generalizes around epoch 1000.

```bash
# Training is done via experiments/scripts/train_spos_wd.py
cd experiments/scripts
python train_spos_wd.py --variant d6_f2_sxyz_rms_wd01 --epochs 3000
```

Training hyperparameters: Adam with weight decay 0.01, lr=1e-3, batch_size=512, 50K training samples per epoch.

## Repository Structure

```
.
├── README.md                 # This file
├── submission_242.py         # Best submission (242 params, 99.07%)
├── submission_260.py         # Runner-up submission (260 params, 100.00%)
├── best_model_242.pt         # Trained weights for 242p model
├── best_model_260.pt         # Trained weights for 260p model
├── adderboard_verify.py      # Official AdderBoard verification script
├── model.py                  # Core model architecture
├── data.py                   # Data generation (LSB-first, fixed padding)
├── config.py                 # Hyperparameters and constants
├── docs/                     # Submission docs + full experiment log
├── experiments/              # Training scripts, ablation code, results
├── checkpoints/              # All experimental checkpoints
└── logs/                     # SLURM training logs
```

## Disclaimer

Code was written with the assistance of Claude Code. The architectural ideas (spiral embeddings, QK-position/V-token split, LSB-first encoding, shared XYZ positions, RMSNorm compression) are original.
