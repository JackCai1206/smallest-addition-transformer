"""
AdderBoard-compatible verification for submission files.

Tests:
1. build_model() returns (model, metadata)
2. Parameter count matches metadata
3. 10 edge cases (max carries, asymmetric, etc.)
4. 10,000 random pairs (seed=2025) for official accuracy
5. 10,000 random pairs (seed=999) for cross-validation
"""

import sys
import time
import random
import importlib.util


def load_submission(path):
    """Import a submission file dynamically."""
    spec = importlib.util.spec_from_file_location("submission", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def count_unique_params(model):
    """Count unique parameters (handles weight tying)."""
    seen = set()
    total = 0
    for name, param in model.named_parameters():
        if param.data_ptr() not in seen:
            seen.add(param.data_ptr())
            total += param.numel()
    return total


def run_edge_cases(mod, model):
    """Test 10 edge cases that stress carry propagation."""
    edge_cases = [
        (9999999999, 9999999999, "max carry chain"),
        (5555555555, 5555555555, "all carries"),
        (1111111111, 8888888888, "sums to 9999999999"),
        (9999999999, 1, "asymmetric max carry"),
        (1, 1, "minimum"),
        (5, 5, "single digit carry"),
        (99, 1, "two digit carry"),
        (999999999, 999999999, "9-digit carry"),
        (1000000000, 9999999999, "mixed magnitudes"),
        (0, 0, "zero + zero"),
    ]

    correct = 0
    total = len(edge_cases)
    for a, b, desc in edge_cases:
        pred = mod.add(model, a, b)
        expected = a + b
        status = "PASS" if pred == expected else "FAIL"
        if pred != expected:
            print(f"  [{status}] {a} + {b} = {pred} (expected {expected}) -- {desc}")
        correct += (pred == expected)

    print(f"Edge cases: {correct}/{total} ({correct/total*100:.1f}%)")
    return correct, total


def run_random_test(mod, model, n=10000, seed=2025):
    """Test on n random pairs with given seed."""
    rng = random.Random(seed)

    correct = 0
    total = n
    bucket_correct = {}
    bucket_total = {}

    for _ in range(n):
        n_digits = rng.randint(1, 10)
        a = rng.randint(0, 10**n_digits - 1)
        b = rng.randint(0, 10**n_digits - 1)
        expected = a + b
        pred = mod.add(model, a, b)

        if pred == expected:
            correct += 1
            bucket_correct[n_digits] = bucket_correct.get(n_digits, 0) + 1
        bucket_total[n_digits] = bucket_total.get(n_digits, 0) + 1

    accuracy = correct / total
    print(f"\nRandom test (n={n}, seed={seed}): {correct}/{total} = {accuracy*100:.2f}%")
    for nd in sorted(bucket_total.keys()):
        bc = bucket_correct.get(nd, 0)
        bt = bucket_total[nd]
        print(f"  {nd:2d}d: {bc:5d}/{bt:5d} = {bc/bt*100:.2f}%")

    return accuracy


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_submission.py <submission_file.py>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"=== Verifying submission: {path} ===\n")

    # 1. Load submission
    print("Loading submission...")
    mod = load_submission(path)
    model, metadata = mod.build_model()
    print(f"  Name: {metadata.get('name', 'N/A')}")
    print(f"  Author: {metadata.get('author', 'N/A')}")
    print(f"  Claimed params: {metadata.get('params', 'N/A')}")

    # 2. Verify parameter count
    unique_params = count_unique_params(model)
    print(f"  Verified unique params: {unique_params}")
    if unique_params != metadata.get('params'):
        print(f"  WARNING: Claimed {metadata['params']} but counted {unique_params}")

    # 3. Edge cases
    print(f"\n--- Edge Cases ---")
    edge_correct, edge_total = run_edge_cases(mod, model)

    # 4. Official AdderBoard test (seed=2025)
    print(f"\n--- Official AdderBoard Test (seed=2025, 10K) ---")
    t0 = time.time()
    official_acc = run_random_test(mod, model, n=10000, seed=2025)
    t1 = time.time()
    print(f"  Time: {t1-t0:.1f}s")

    # 5. Cross-validation (seed=999)
    print(f"\n--- Cross-validation (seed=999, 10K) ---")
    cross_acc = run_random_test(mod, model, n=10000, seed=999)

    # Summary
    print(f"\n{'='*50}")
    print(f"SUMMARY: {path}")
    print(f"  Unique parameters: {unique_params}")
    print(f"  Edge cases: {edge_correct}/{edge_total}")
    print(f"  AdderBoard accuracy (seed=2025): {official_acc*100:.2f}%")
    print(f"  Cross-val accuracy (seed=999): {cross_acc*100:.2f}%")
    print(f"  PASSES AdderBoard (>=99%): {'YES' if official_acc >= 0.99 else 'NO'}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
