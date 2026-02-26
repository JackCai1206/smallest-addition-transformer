"""Quick evaluation of best models on 10K AdderBoard test sets."""
import sys, torch
from torch.utils.data import DataLoader
from collections import defaultdict
from data import AdditionDataset
from config import BATCH_SIZE, MAX_DIGITS, EQ_POS, ANSWER_LEN

def evaluate(model, dataloader, device='cpu'):
    model.eval()
    total_exact = total_seq = 0
    bucket_correct = defaultdict(int)
    bucket_total = defaultdict(int)
    with torch.no_grad():
        for tokens, loss_mask, n_digits in dataloader:
            tokens = tokens.to(device)
            prompt = tokens[:, :EQ_POS + 1]
            answer_target = tokens[:, EQ_POS + 1:]
            generated = model.generate(prompt)
            exact = (generated == answer_target).all(dim=-1)
            total_exact += exact.sum().item()
            total_seq += tokens.size(0)
            for nd in range(1, MAX_DIGITS + 1):
                mask_nd = (n_digits == nd)
                if mask_nd.any():
                    bucket_total[nd] += mask_nd.sum().item()
                    bucket_correct[nd] += (exact & mask_nd.to(device)).sum().item()
    ea = total_exact / max(total_seq, 1)
    buckets = {nd: bucket_correct[nd] / max(bucket_total[nd], 1) for nd in sorted(bucket_total.keys())}
    return ea, buckets

if __name__ == '__main__':
    model_type = sys.argv[1] if len(sys.argv) > 1 else 'tp_f3'

    if model_type == 'tp_f3':
        from train_winner import WinnerModel
        model = WinnerModel(no_bias=False, ffn_dim=3, factored_pos=False, norm_type='ln')
        ckpt = 'best_winner_tp_f3.pt'
    elif model_type == 'tp_f3_push':
        from train_push import PushModel
        model = PushModel(ffn_dim=3, head_dim=3, n_heads=2)
        ckpt = 'best_push_tp_f3_ref.pt'
    else:
        print(f"Unknown model type: {model_type}")
        sys.exit(1)

    n_train, n_total = model.count_parameters()
    print(f"Model: {model_type}, Params: {n_train} trainable / {n_total} total")

    model.load_state_dict(torch.load(ckpt, weights_only=True, map_location='cpu'))

    for seed_name, seed in [("seed=999", 999), ("seed=2025", 2025)]:
        ds = AdditionDataset(10000, seed=seed)
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
        ea, buckets = evaluate(model, dl)
        bucket_str = " ".join(f"{nd}d:{acc:.4f}" for nd, acc in buckets.items())
        print(f"10K ({seed_name}): ea={ea:.4f} | {bucket_str}")
