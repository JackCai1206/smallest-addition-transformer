"""Training loop for QK=pos, V=tok variant."""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict

try:
    import wandb
except ImportError:
    wandb = None

from config import (
    BATCH_SIZE, LR, WEIGHT_DECAY, EPOCHS, VAL_SIZE,
    TRAIN_SAMPLES_PER_EPOCH, LOG_EVERY, PATIENCE,
    D_MODEL, TOK_DIM, POS_DIM, N_POS, N_HEADS, N_LAYERS, FFN_DIM,
    MAX_SEQ_LEN, MAX_DIGITS, EQ_POS, ANSWER_LEN, EOS_TOKEN
)
from data import AdditionDataset
from model_qkpos_vtok import SmallestAdditionTransformerQKPosVTok


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    total_exact = 0
    total_sequences = 0
    bucket_correct = defaultdict(int)
    bucket_total = defaultdict(int)

    with torch.no_grad():
        for tokens, loss_mask, n_digits in dataloader:
            tokens = tokens.to(device)
            loss_mask = loss_mask.to(device)

            logits = model(tokens[:, :-1])
            targets = tokens[:, 1:]
            mask = loss_mask[:, 1:]

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction='none'
            ).reshape(targets.shape)

            masked_loss = (loss * mask).sum()
            n_tok = mask.sum()
            if n_tok > 0:
                total_loss += masked_loss.item()
                total_tokens += n_tok.item()

            preds_tf = logits.argmax(dim=-1)
            correct = (preds_tf == targets) & (mask == 1)
            total_correct += correct.sum().item()

            prompt = tokens[:, :EQ_POS + 1]
            answer_target = tokens[:, EQ_POS + 1:]
            generated = model.generate(prompt)

            exact = (generated == answer_target).all(dim=-1)
            total_exact += exact.sum().item()
            total_sequences += tokens.size(0)

            for nd in range(1, MAX_DIGITS + 1):
                mask_nd = (n_digits == nd)
                if mask_nd.any():
                    bucket_total[nd] += mask_nd.sum().item()
                    bucket_correct[nd] += (exact & mask_nd.to(device)).sum().item()

    avg_loss = total_loss / max(total_tokens, 1)
    digit_acc = total_correct / max(total_tokens, 1)
    exact_acc = total_exact / max(total_sequences, 1)
    bucket_acc = {
        nd: bucket_correct[nd] / max(bucket_total[nd], 1)
        for nd in sorted(bucket_total.keys())
    }

    return avg_loss, digit_acc, exact_acc, bucket_acc


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = SmallestAdditionTransformerQKPosVTok().to(device)
    n_trainable, n_total = model.count_parameters()
    print(f"Model parameters: {n_trainable} trainable / {n_total} total")
    print(f"Model architecture:\n{model}\n")

    use_wandb = (
        wandb is not None
        and os.environ.get("WANDB_MODE") != "disabled"
    )
    if use_wandb:
        wandb.init(
            project="smallest-addition-tf",
            config={
                "variant": "qkpos_vtok_2layer_nomlp",
                "d_model": D_MODEL,
                "tok_dim": TOK_DIM,
                "pos_dim": POS_DIM,
                "n_pos": N_POS,
                "n_heads": N_HEADS,
                "optimizer": "Adam",
                "n_layers": N_LAYERS,
                "ffn_dim": 0,
                "max_seq_len": MAX_SEQ_LEN,
                "max_digits": MAX_DIGITS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "epochs": EPOCHS,
                "train_samples_per_epoch": TRAIN_SAMPLES_PER_EPOCH,
                "val_size": VAL_SIZE,
                "n_params_trainable": n_trainable,
                "n_params_total": n_total,
                "attention": "QK=pos, V=tok",
            },
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    val_dataset = AdditionDataset(VAL_SIZE, seed=42)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    best_exact_acc = -1.0
    best_epoch = 0
    patience_counter = 0
    global_step = 0

    for epoch in range(1, EPOCHS + 1):
        train_dataset = AdditionDataset(TRAIN_SAMPLES_PER_EPOCH, seed=None)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        batch_count = 0

        for batch_idx, (tokens, loss_mask, _n_digits) in enumerate(train_loader):
            tokens = tokens.to(device)
            loss_mask = loss_mask.to(device)

            logits = model(tokens[:, :-1])
            targets = tokens[:, 1:]
            mask = loss_mask[:, 1:]

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction='none'
            ).reshape(targets.shape)

            masked_loss = (loss * mask).sum()
            n_tokens = mask.sum()

            if n_tokens > 0:
                avg_loss = masked_loss / n_tokens
                optimizer.zero_grad()
                avg_loss.backward()
                optimizer.step()

                epoch_loss += masked_loss.item()
                epoch_tokens += n_tokens.item()

                global_step += 1
                if use_wandb:
                    wandb.log({"train/batch_loss": avg_loss.item(), "global_step": global_step})

            batch_count += 1
            if batch_count % LOG_EVERY == 0:
                running_loss = epoch_loss / max(epoch_tokens, 1)
                print(f"  Epoch {epoch} batch {batch_count}: train_loss={running_loss:.4f}")

        train_avg_loss = epoch_loss / max(epoch_tokens, 1)

        val_loss, val_digit_acc, val_exact_acc, bucket_acc = evaluate(model, val_loader, device)

        bucket_str = " ".join(f"{nd}d:{acc:.3f}" for nd, acc in bucket_acc.items())

        print(
            f"Epoch {epoch:3d} | "
            f"train_loss={train_avg_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"digit_acc={val_digit_acc:.4f} | "
            f"exact_acc={val_exact_acc:.4f} | "
            f"by_len: {bucket_str}"
        )

        if use_wandb:
            log_dict = {
                "epoch": epoch,
                "train/epoch_loss": train_avg_loss,
                "val/loss": val_loss,
                "val/digit_acc": val_digit_acc,
                "val/exact_acc": val_exact_acc,
            }
            for nd, acc in bucket_acc.items():
                log_dict[f"val/exact_acc_{nd}d"] = acc
            wandb.log(log_dict)

        if val_exact_acc > best_exact_acc:
            best_exact_acc = val_exact_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model_qkpos_vtok.pt')
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}. Best exact_acc={best_exact_acc:.4f} at epoch {best_epoch}")
            break

    print(f"\nTraining complete. Best val_exact_acc={best_exact_acc:.4f} at epoch {best_epoch}")

    model.load_state_dict(torch.load('best_model_qkpos_vtok.pt', weights_only=True))
    val_loss, val_digit_acc, val_exact_acc, bucket_acc = evaluate(model, val_loader, device)
    bucket_str = " ".join(f"{nd}d:{acc:.3f}" for nd, acc in bucket_acc.items())
    print(f"Final eval | val_loss={val_loss:.4f} | digit_acc={val_digit_acc:.4f} | exact_acc={val_exact_acc:.4f}")
    print(f"Final by_len: {bucket_str}")

    if use_wandb:
        log_dict = {
            "final/val_loss": val_loss,
            "final/val_digit_acc": val_digit_acc,
            "final/val_exact_acc": val_exact_acc,
            "final/best_epoch": best_epoch,
        }
        for nd, acc in bucket_acc.items():
            log_dict[f"final/exact_acc_{nd}d"] = acc
        wandb.log(log_dict)
        wandb.finish()


if __name__ == '__main__':
    train()
