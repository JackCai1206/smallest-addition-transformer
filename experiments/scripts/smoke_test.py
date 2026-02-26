"""Quick smoke test."""
from model import SmallestAdditionTransformer, build_token_embedding, build_positional_encoding
from data import AdditionDataset
from torch.utils.data import DataLoader

model = SmallestAdditionTransformer()
trainable, total = model.count_parameters()
print(f'Parameters: {trainable} trainable / {total} total')
for name, p in model.named_parameters():
    print(f'  {name:30s} {str(list(p.shape)):20s} = {p.numel()}')

print(f'\nToken embedding (frozen buffer, {model.tok_emb.shape}):')
te = build_token_embedding()
for i in range(14):
    label = str(i) if i < 10 else ['+', '=', 'EOS', 'PAD'][i - 10]
    print(f'  {label:>4s}: [{", ".join(f"{v:.3f}" for v in te[i].tolist())}]')

print(f'\nPos encoding (frozen buffer, {model.pos_enc.shape})')

print(f'\nAttention: Q,K from pos dims (last 3), V from tok dims (first 3)')

ds = AdditionDataset(8, seed=0)
loader = DataLoader(ds, batch_size=8)
tokens, mask, n_digits = next(iter(loader))
print(f'\nn_digits: {n_digits.tolist()}')

tok = tokens[0].tolist()
x_digits, y_digits, z_digits = tok[:10], tok[11:21], tok[22:33]
x = sum(d * 10**i for i, d in enumerate(x_digits))
y = sum(d * 10**i for i, d in enumerate(y_digits))
z = sum(d * 10**i for i, d in enumerate(z_digits))
print(f'Sample: {x} + {y} = {z} (correct: {x+y==z}, n_digits={n_digits[0].item()})')

prompt = tokens[:, :22]
gen = model.generate(prompt)
print(f'Generated shape: {gen.shape} (expect [8, 12])')
print('Smoke test passed!')
