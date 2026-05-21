from __future__ import annotations

import argparse
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset import get_data_loaders
from model import HandSignTransformer


def plot_loss(history: dict, save_path: str = "loss.png") -> None:
    epochs = range(1, len(history["cost"]) + 1)
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color_cost = "#005A96"
    color_error = "#FF6666"

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cost (cross-entropy loss)", color=color_cost)
    ax1.plot(epochs, history["cost"], color=color_cost, label="Cost")
    ax1.tick_params(axis="y", labelcolor=color_cost)
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Error rate")
    ax2.plot(
        epochs,
        history["train_error"],
        color=color_error,
        linestyle="-",
        label="Train error",
    )
    ax2.plot(
        epochs,
        history["valid_error"],
        color=color_error,
        linestyle="--",
        label="Valid error",
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    classes: list[str],
    hparams: dict[str, Any],
) -> None:
    torch.save({"state_dict": model.state_dict(), "classes": classes, **hparams}, path)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    bar = tqdm(
        loader, desc="  train" if training else "    val", leave=False, position=1
    )

    with torch.set_grad_enabled(training):
        for inputs, labels in bar:
            inputs, labels = inputs.to(device), labels.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            correct += outputs.argmax(1).eq(labels).sum().item()
            total += inputs.size(0)
            bar.set_postfix(
                loss=f"{total_loss / total:.4f}", acc=f"{correct / total:.2%}"
            )

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train HandSignTransformer")
    parser.add_argument("--data", type=str, default="data/npz")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)

    parser.add_argument("--output", "-o", type=str, default="best.pth")

    # transformer params
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print(f"[*] Loading dataset from {args.data} ...")
    train_loader, val_loader, _, classes = get_data_loaders(
        root=args.data,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(f"[*] Classes: {classes}")
    print(f"[*] train: {len(train_loader.dataset)}  val: {len(val_loader.dataset)}")  # type: ignore

    hparams = {
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "dim_feedforward": args.dim_feedforward,
    }
    model = HandSignTransformer(num_classes=len(classes), **hparams).to(device)

    criterion = nn.CrossEntropyLoss(reduction="sum")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    history: dict[str, list] = {"cost": [], "train_error": [], "valid_error": []}

    print("[*] Start Training ...")
    epoch_bar = tqdm(range(1, args.epochs + 1), desc="Epochs", position=0)
    for epoch in epoch_bar:
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["cost"].append(train_loss)
        history["train_error"].append(1.0 - train_acc)
        history["valid_error"].append(1.0 - val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                path=args.output, model=model, classes=classes, hparams=hparams
            )

        epoch_bar.set_description(
            f"Epoch {epoch:>2} | "
            f"train {train_loss:.4f}/{train_acc:.2%} | "
            f"val {val_loss:.4f}/{val_acc:.2%}"
        )

        plot_loss(history)

    print(f"\n[+] Training done. Best val acc: {best_val_acc:.2%}")
    print(f"[+] Checkpoint saved → {args.output}")


if __name__ == "__main__":
    main()
