"""d5 experiments with weight decay and learning rate scheduling.

Weight decay can help grokking by encouraging the model to find
simpler solutions. LR scheduling (warmup + cosine) is standard.
"""

import argparse, math, os, json
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict
from data import AdditionDataset
from config import (
    VOCAB_SIZE, MAX_SEQ_LEN, MAX_DIGITS, ANSWER_LEN, EQ_POS,
    BATCH_SIZE, VAL_SIZE, TRAIN_SAMPLES_PER_EPOCH,
)
from train_d5 import D5Model, evaluate


VARIANTS = {
    # d5_f3 with weight decay (most promising size)
    'd5_f3_wd01': dict(
        model=dict(tok_dim=2, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=3),
        lr=1e-3, wd=0.01, schedule='none'),
    'd5_f3_wd1': dict(
        model=dict(tok_dim=2, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=3),
        lr=1e-3, wd=0.1, schedule='none'),
    'd5_f3_cos': dict(
        model=dict(tok_dim=2, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=3),
        lr=1e-3, wd=0.0, schedule='cosine'),
    'd5_f3_wd_cos': dict(
        model=dict(tok_dim=2, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=3),
        lr=1e-3, wd=0.01, schedule='cosine'),

    # d5_f4 with weight decay
    'd5_f4_wd01': dict(
        model=dict(tok_dim=2, pos_dim=3, ffn_dim=4, n_heads=2, head_dim=3),
        lr=1e-3, wd=0.01, schedule='none'),
    'd5_f4_wd_cos': dict(
        model=dict(tok_dim=2, pos_dim=3, ffn_dim=4, n_heads=2, head_dim=3),
        lr=1e-3, wd=0.01, schedule='cosine'),

    # tp_f3 (d6) with weight decay — see if WD helps the d6 model too
    'd6_f3_wd01': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=3),
        lr=1e-3, wd=0.01, schedule='none'),
    'd6_f3_wd1': dict(
        model=dict(tok_dim=3, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=3),
        lr=1e-3, wd=0.1, schedule='none'),

    # Higher LR for d5
    'd5_f3_lr3': dict(
        model=dict(tok_dim=2, pos_dim=3, ffn_dim=3, n_heads=2, head_dim=3),
        lr=3e-3, wd=0.0, schedule='none'),
    'd5_f4_lr3': dict(
        model=dict(tok_dim=2, pos_dim=3, ffn_dim=4, n_heads=2, head_dim=3),
        lr=3e-3, wd=0.0, schedule='none'),
}


def train(variant_name):
    cfg = VARIANTS[variant_name]
    model_cfg = cfg['model']
    lr = cfg['lr']
    wd = cfg['wd']
    schedule = cfg['schedule']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Variant: {variant_name}")
    print(f"Model: {model_cfg}")
    print(f"LR: {lr}, WD: {wd}, Schedule: {schedule}")

    model = D5Model(**model_cfg).to(device)
    n_trainable, n_total = model.count_parameters()
    print(f"Parameters: {n_trainable} trainable / {n_total} total")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    max_epochs = 1000
    if schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)
    else:
        scheduler = None

    val_dataset = AdditionDataset(VAL_SIZE, seed=42)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    best_exact_acc = -1.0
    best_epoch = 0
    patience_counter = 0
    patience = 500
    save_name = f'best_d5wd_{variant_name}.pt'

    for epoch in range(1, max_epochs + 1):
        train_dataset = AdditionDataset(TRAIN_SAMPLES_PER_EPOCH, seed=None)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        model.train()
        epoch_loss = epoch_tokens = 0
        for tokens, loss_mask, _nd in train_loader:
            tokens, loss_mask = tokens.to(device), loss_mask.to(device)
            logits = model(tokens[:, :-1])
            targets = tokens[:, 1:]
            mask = loss_mask[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                                   reduction='none').reshape(targets.shape)
            masked_loss = (loss * mask).sum()
            n_tokens = mask.sum()
            if n_tokens > 0:
                (masked_loss / n_tokens).backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += masked_loss.item()
                epoch_tokens += n_tokens.item()

        if scheduler:
            scheduler.step()

        val_loss, val_da, val_ea, bucket = evaluate(model, val_loader, device)
        bucket_str = " ".join(f"{nd}d:{acc:.3f}" for nd, acc in bucket.items())
        curr_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d} | lr={curr_lr:.6f} | train_loss={epoch_loss/max(epoch_tokens,1):.4f} | "
              f"val_loss={val_loss:.4f} | da={val_da:.4f} | ea={val_ea:.4f} | {bucket_str}")

        if val_ea > best_exact_acc:
            best_exact_acc = val_ea
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), save_name)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stop at epoch {epoch}. Best ea={best_exact_acc:.4f} at epoch {best_epoch}")
            break

    print(f"Training done. Best ea={best_exact_acc:.4f} at epoch {best_epoch}")

    if os.path.exists(save_name):
        model.load_state_dict(torch.load(save_name, weights_only=True))
    for seed_name, seed in [("seed=999", 999), ("seed=2025", 2025)]:
        _, _, final_ea, final_bucket = evaluate(model,
            DataLoader(AdditionDataset(10000, seed=seed), batch_size=BATCH_SIZE), device)
        bucket_str = " ".join(f"{nd}d:{acc:.3f}" for nd, acc in final_bucket.items())
        print(f"Final (10K, {seed_name}): ea={final_ea:.4f} | {bucket_str}")

    with open(f'results_d5wd_{variant_name}.json', 'w') as f:
        json.dump({'variant': variant_name, 'params': n_total, 'best_ea': best_exact_acc,
                   'best_epoch': best_epoch, 'lr': lr, 'wd': wd, 'schedule': schedule}, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, required=True, choices=list(VARIANTS.keys()))
    args = parser.parse_args()
    train(args.variant)


if __name__ == '__main__':
    main()
