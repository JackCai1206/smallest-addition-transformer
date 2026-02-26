"""Data generation and dataset class for integer addition."""

import numpy as np
import torch
from torch.utils.data import Dataset

from config import (
    MAX_DIGITS, PLUS_TOKEN, EQUALS_TOKEN, EOS_TOKEN, MAX_SEQ_LEN,
    EQ_POS
)

# Precompute the fixed loss mask (same for every sample)
_PROMPT_LEN = MAX_DIGITS + 1 + MAX_DIGITS + 1  # 22
_ANSWER_LEN = MAX_DIGITS + 1 + 1               # 12
_LOSS_MASK = np.array([0] * _PROMPT_LEN + [1] * _ANSWER_LEN, dtype=np.int64)


def generate_batch_np(num_samples, rng):
    """Generate num_samples addition examples.

    Returns:
        tokens: (num_samples, MAX_SEQ_LEN) int64 array
        n_digits: (num_samples,) int64 array — number of digits in operands
    """
    # Pick random digit-count per sample (1..MAX_DIGITS)
    n = rng.integers(1, MAX_DIGITS + 1, size=num_samples)

    # For each sample, sample x and y uniformly from [lo, hi)
    lo = np.where(n == 1, 0, 10 ** (n - 1))
    hi = 10 ** n  # exclusive upper bound
    x = (rng.random(num_samples) * (hi - lo) + lo).astype(np.int64)
    y = (rng.random(num_samples) * (hi - lo) + lo).astype(np.int64)
    z = x + y

    # Build token array (num_samples, 34)
    tokens = np.empty((num_samples, MAX_SEQ_LEN), dtype=np.int64)

    # Fill x digits (positions 0..9) — LSB first, zero-padded to MAX_DIGITS
    tmp = x.copy()
    for d in range(MAX_DIGITS):
        tokens[:, d] = tmp % 10
        tmp //= 10

    # Plus token at position 10
    tokens[:, MAX_DIGITS] = PLUS_TOKEN

    # Fill y digits (positions 11..20)
    tmp = y.copy()
    for d in range(MAX_DIGITS):
        tokens[:, MAX_DIGITS + 1 + d] = tmp % 10
        tmp //= 10

    # Equals token at position 21
    tokens[:, 2 * MAX_DIGITS + 1] = EQUALS_TOKEN

    # Fill z digits (positions 22..32) — padded to MAX_DIGITS+1
    tmp = z.copy()
    for d in range(MAX_DIGITS + 1):
        tokens[:, 2 * MAX_DIGITS + 2 + d] = tmp % 10
        tmp //= 10

    # EOS at position 33
    tokens[:, -1] = EOS_TOKEN

    return tokens, n


class AdditionDataset(Dataset):
    """Dataset of fixed addition samples (generated at init)."""

    def __init__(self, num_samples, seed=None):
        rng = np.random.default_rng(seed)
        tokens_np, n_digits_np = generate_batch_np(num_samples, rng)
        self.tokens = torch.from_numpy(tokens_np)
        self.loss_mask = torch.from_numpy(
            np.tile(_LOSS_MASK, (num_samples, 1))
        )
        self.n_digits = torch.from_numpy(n_digits_np)  # digit length per sample

    def __len__(self):
        return self.tokens.shape[0]

    def __getitem__(self, idx):
        return self.tokens[idx], self.loss_mask[idx], self.n_digits[idx]
