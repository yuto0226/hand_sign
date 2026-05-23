"""Three-phase surgical fine-tuning for EfficientNet-B0.

Usage:
    python -m cnnclassifier.train --root data/cnn --train-sessions s1 --test-sessions s2
"""

from __future__ import annotations

import argparse
from typing import cast

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from cnnclassifier.dataset import make_splits
from cnnclassifier.model import build_model, freeze_backbone, unfreeze_blocks


def cutmix_batch(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    B, C, H, W = x.shape
    perm = torch.randperm(B, device=x.device)

    cut_h = int(H * (1 - lam) ** 0.5)
    cut_w = int(W * (1 - lam) ** 0.5)
    cx = int(torch.randint(W, (1,)))
    cy = int(torch.randint(H, (1,)))
    x1 = max(0, cx - cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    x2 = min(W, cx + cut_w // 2)
    y2 = min(H, cy + cut_h // 2)

    mixed = x.clone()
    mixed[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
    lam_actual = 1.0 - (y2 - y1) * (x2 - x1) / (H * W)
    return mixed, y, y[perm], lam_actual


def _train_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cutmix_p: float = 0.3,
) -> tuple[float, float]:
    model.train()
    total_loss = correct = total = 0
    for x, y in tqdm(loader, desc="  train", leave=False):
        x, y = x.to(device), y.to(device)
        if torch.rand(1).item() < cutmix_p:
            x, ya, yb, lam = cutmix_batch(x, y)
            logits = model(x)
            loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
            pred = logits.argmax(1)
            correct += (
                (lam * (pred == ya).float() + (1 - lam) * (pred == yb).float())
                .sum()
                .item()
            )
        else:
            logits = model(x)
            loss = criterion(logits, y)
            correct += (logits.argmax(1) == y).sum().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        total += len(y)
    return total_loss / total, correct / total


@torch.no_grad()
def _val_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = correct = total = 0
    for x, y in tqdm(loader, desc="    val", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return total_loss / total, correct / total


def train(
    root: str = "data/cnn",
    train_sessions: list[str] | None = None,
    test_sessions: list[str] | None = None,
    batch_size: int = 32,
    save_path: str = "best_cnn.pth",
    seed: int = 42,
) -> None:
    if train_sessions is None:
        train_sessions = ["s1"]
    if test_sessions is None:
        test_sessions = ["s2"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader, _, classes = make_splits(
        root,
        train_sessions=train_sessions,
        test_sessions=test_sessions,
        batch_size=batch_size,
        seed=seed,
    )
    print(f"[*] classes ({len(classes)}): {classes}")

    model = build_model(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    print("[*] head only")
    freeze_backbone(model)
    classifier_p1 = cast(nn.Sequential, getattr(model, "classifier", None))
    opt = Adam(classifier_p1.parameters(), lr=1e-3)
    for epoch in tqdm(range(10), desc="head only"):
        tl, ta = _train_epoch(model, train_loader, criterion, opt, device, cutmix_p=0.0)
        vl, va = _val_epoch(model, val_loader, criterion, device)
        tqdm.write(f"  [{epoch + 1:02d}] train acc={ta:.3f}  val acc={va:.3f}")

    print("[*] last 2 blocks")
    unfreeze_blocks(model, [7, 8])
    features = cast(nn.Sequential, getattr(model, "features", None))
    classifier = cast(nn.Sequential, getattr(model, "classifier", None))
    opt = Adam(
        [
            {"params": features[7].parameters(), "lr": 1e-4},
            {"params": features[8].parameters(), "lr": 1e-4},
            {"params": classifier.parameters(), "lr": 1e-4},
        ]
    )
    scheduler = CosineAnnealingWarmRestarts(opt, T_0=10)
    best_val_loss = float("inf")

    for epoch in tqdm(range(15), desc="last 2 blocks"):
        tl, ta = _train_epoch(model, train_loader, criterion, opt, device)
        vl, va = _val_epoch(model, val_loader, criterion, device)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        tqdm.write(
            f"  [{epoch + 1:02d}] train acc={ta:.3f}  val acc={va:.3f}  lr={lr_now:.2e}"
        )
        if vl < best_val_loss:
            best_val_loss = vl
            torch.save({"model": model.state_dict(), "classes": classes}, save_path)
            tqdm.write(f"       ↳ saved {save_path}")

    print(f"\n[*] best checkpoint → {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/cnn")
    parser.add_argument("--train-sessions", nargs="+", default=["s1"])
    parser.add_argument("--test-sessions", nargs="+", default=["s2"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save", default="best_cnn.pth")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(
        root=args.root,
        train_sessions=args.train_sessions,
        test_sessions=args.test_sessions,
        batch_size=args.batch_size,
        save_path=args.save,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
