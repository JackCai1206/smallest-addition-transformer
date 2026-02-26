"""Independent verification of the trained model on fresh random data."""

import torch
from data import AdditionDataset
from torch.utils.data import DataLoader
from model_trainable_emb import SmallestAdditionTransformerTrainable
from config import EQ_POS, MAX_DIGITS
from collections import defaultdict
import random

model = SmallestAdditionTransformerTrainable()
model.load_state_dict(torch.load('best_model_trainable_emb.pt', weights_only=True))
model.eval()

# Test on 10,000 fresh samples with a different seed
for test_seed in [999, 12345, 77777]:
    ds = AdditionDataset(10000, seed=test_seed)
    loader = DataLoader(ds, batch_size=256, shuffle=False)

    total_exact = 0
    total = 0
    bucket_correct = defaultdict(int)
    bucket_total = defaultdict(int)
    failures = []

    with torch.no_grad():
        for tokens, loss_mask, n_digits in loader:
            prompt = tokens[:, :EQ_POS + 1]
            answer_target = tokens[:, EQ_POS + 1:]
            generated = model.generate(prompt)

            exact = (generated == answer_target).all(dim=-1)
            total_exact += exact.sum().item()
            total += tokens.size(0)

            for nd in range(1, MAX_DIGITS + 1):
                mask_nd = (n_digits == nd)
                if mask_nd.any():
                    bucket_total[nd] += mask_nd.sum().item()
                    bucket_correct[nd] += (exact & mask_nd).sum().item()

            # Collect some failures for inspection
            if not exact.all() and len(failures) < 10:
                for i in range(tokens.size(0)):
                    if not exact[i] and len(failures) < 10:
                        failures.append((tokens[i].tolist(), generated[i].tolist(), answer_target[i].tolist(), n_digits[i].item()))

    print(f"\n=== Seed {test_seed}: {total_exact}/{total} = {total_exact/total:.4f} exact match ===")
    for nd in sorted(bucket_total.keys()):
        acc = bucket_correct[nd] / max(bucket_total[nd], 1)
        print(f"  {nd:2d}d: {bucket_correct[nd]:5d}/{bucket_total[nd]:5d} = {acc:.4f}")

    if failures:
        print(f"\nSample failures:")
        for tok, gen, tgt, nd in failures[:5]:
            x_digits = tok[:10]
            y_digits = tok[11:21]
            x = sum(d * 10**i for i, d in enumerate(x_digits) if d < 10)
            y = sum(d * 10**i for i, d in enumerate(y_digits) if d < 10)
            z_pred = sum(d * 10**i for i, d in enumerate(gen) if d < 10)
            z_true = sum(d * 10**i for i, d in enumerate(tgt) if d < 10)
            print(f"  {nd}d: {x} + {y} = {z_pred} (expected {z_true})")
    else:
        print("  No failures!")

# Also test some specific hard cases manually
print("\n=== Manual spot checks ===")
from data import generate_batch_np
import numpy as np

# Known hard cases: lots of carries
hard_cases = [
    (9999999999, 9999999999),  # max carry chain
    (5555555555, 5555555555),  # all carries
    (1111111111, 8888888888),  # 9999999999
    (9999999999, 1),           # asymmetric
    (0, 0),                    # edge: but our data has 1-digit min
    (1, 1),
    (5, 5),
    (99, 1),
    (999999999, 999999999),
]

for x, y in hard_cases:
    # Build token sequence manually
    nd = max(len(str(x)), len(str(y)))
    x_digits = [(x // 10**i) % 10 for i in range(10)]
    y_digits = [(y // 10**i) % 10 for i in range(10)]
    z_true = x + y

    tokens = x_digits + [10] + y_digits + [11]  # +, =
    tokens = tokens + [0] * 12  # placeholder answer + EOS
    tokens = torch.tensor([tokens], dtype=torch.long)

    prompt = tokens[:, :EQ_POS + 1]
    with torch.no_grad():
        gen = model.generate(prompt)

    gen_digits = gen[0].tolist()
    z_pred = sum(d * 10**i for i, d in enumerate(gen_digits) if d < 10)
    status = "OK" if z_pred == z_true else "FAIL"
    print(f"  {x} + {y} = {z_pred} (expected {z_true}) [{status}]")
