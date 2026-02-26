"""Test AdderBoard edge cases."""
from submission import build_model, add

model, _ = build_model()
cases = [
    (0, 0),
    (0, 1),
    (9_999_999_999, 0),
    (9_999_999_999, 1),
    (9_999_999_999, 9_999_999_999),
    (5_000_000_000, 5_000_000_000),
    (1_111_111_111, 8_888_888_889),
    (1_234_567_890, 9_876_543_210),
    (9_999_999_999, 9_999_999_999),
    (1, 9_999_999_999),
]
for a, b in cases:
    r = add(model, a, b)
    ok = 'OK' if r == a + b else 'FAIL'
    print(f'{a:>13d} + {b:>13d} = {r:>14d} (expected {a+b:>14d}) [{ok}]')
