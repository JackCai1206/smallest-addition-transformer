# Spiral Addition Transformer

A **438-parameter** transformer that learns 10-digit integer addition with **99.83% accuracy** — and converges in just ~75 epochs with no grokking.

## Key Results

- **Parameters**: 438 (all trainable)
- **Accuracy**: 99.83% on AdderBoard verification (10,010 test cases, seed=2025)
- **Convergence**: ~75 epochs (~3.75M training samples) to 100% on validation
- **No grokking**: Spiral initialization of embeddings eliminates the delayed generalization phase

## Architecture

```
1-layer decoder-only transformer
  d_model = 6 = tok_dim (3) + pos_dim (3)
  2 attention heads, head_dim = 3
  FFN inner dim = 6 (1x expansion)

  Token embedding: 3D spiral over digit value (trainable, spiral-initialized)
  Positional encoding: 3D spiral over digit position (trainable, spiral-initialized)

  Attention routing:
    Q, K from positional dims (last 3) — attention patterns are position-based
    V from token dims (first 3) — values carry digit information

  Input format: LSB-first digits, zero-padded to fixed length
    e.g., 356 + 478 = 834 → "6 5 3 0...0 + 8 7 4 0...0 = 4 3 8 0...0 EOS"
```

## Why No Grokking?

The spiral initialization gives the model a structured starting point:
- **Token embeddings** are initialized as a 3D spiral over digits 0-9 (cos, sin, linear ramp), providing a smooth, geometrically meaningful representation
- **Positional encodings** use the same spiral pattern, with matching positions across the two operands and the sum aligned to identical encodings

This means the model starts with embeddings that already encode useful structure, rather than learning it from scratch. Combined with the QK/V split (attention routes by position, values carry digit info), the model can learn addition rapidly without the "memorize then generalize" grokking phase.

## Usage

```bash
# Verify with AdderBoard
python verify.py submission.py

# Train from scratch
python train_trainable_emb.py

# Quick smoke test
python smoke_test.py
```

## AdderBoard Verification Output

```
Model: Spiral Addition Transformer
Author: Jack Cai
Parameters (unique): 438
Architecture: 1L decoder, d=6 (3 tok + 3 pos), 2h, hd=3, ff=6

Results: 9993/10010 correct (99.83%)
Time: 59.0s (170 additions/sec)
Status: QUALIFIED (threshold: 99%)
```

## Disclaimer

Code was written with the assistance of Claude Code. The architectural ideas (spiral embeddings, QK-position/V-token split, LSB-first encoding) are original.

## Files

| File | Description |
|------|-------------|
| `submission.py` | AdderBoard submission (defines `build_model()` and `add()`) |
| `model.py` | Core model with spiral embeddings and split attention |
| `model_trainable_emb.py` | Trainable-embedding variant (the winning config) |
| `train_trainable_emb.py` | Training loop for the trainable variant |
| `data.py` | Vectorized data generation (LSB-first, fixed padding) |
| `config.py` | All hyperparameters and constants |
| `best_model_trainable_emb.pt` | Trained weights (438 params) |
| `verify.py` | AdderBoard's official verification script |
