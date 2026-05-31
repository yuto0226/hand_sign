"""Evaluate a trained CNN checkpoint on the test split.

Usage:
    python -m cnnclassifier.evaluate --checkpoint best_cnn.pth
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from cnnclassifier.dataset import make_splits
from cnnclassifier.model import build_model


def load_checkpoint(path: str, device: torch.device) -> tuple[dict, list[str]]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    return ckpt["model"], ckpt["classes"]


@torch.no_grad()
def evaluate(
    checkpoint: str = "best_cnn.pth",
    root: str = "data/cnn",
    train_sessions: list[str] | None = None,
    test_sessions: list[str] | None = None,
    seed: int = 42,
) -> None:
    if train_sessions is None:
        train_sessions = ["s1"]
    if test_sessions is None:
        test_sessions = ["s2"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict, classes = load_checkpoint(checkpoint, device)

    model = build_model(num_classes=len(classes)).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    _, _, test_loader, dataset_classes = make_splits(
        root,
        train_sessions=train_sessions,
        test_sessions=test_sessions,
        num_workers=0,
        seed=seed,
    )

    if dataset_classes != classes:
        raise ValueError(
            "Dataset classes do not match checkpoint classes. "
            f"dataset={dataset_classes} checkpoint={classes}"
        )

    all_preds: list[int] = []
    all_labels: list[int] = []
    latencies: list[float] = []

    for x, y in test_loader:
        x = x.to(device)
        t0 = time.perf_counter()
        logits = model(x)
        ms_per_sample = (time.perf_counter() - t0) / len(x) * 1000
        latencies.append(ms_per_sample)
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(y.tolist())

    if not all_labels:
        raise SystemExit(
            "Test split is empty. Your dataset likely has no files matching "
            f"test_sessions={test_sessions}. "
            "Either record that session (e.g. s2_*.jpg) or pass a different "
            "--test-sessions value."
        )

    labels = list(range(len(classes)))
    print(
        classification_report(
            all_labels,
            all_preds,
            labels=labels,
            target_names=classes,
            digits=3,
            zero_division="warn",
        )
    )
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(all_labels, all_preds, labels=labels))
    print(f"\nAvg inference latency: {np.mean(latencies):.2f} ms/sample")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="best_cnn.pth")
    parser.add_argument("--root", default="data/cnn")
    parser.add_argument("--train-sessions", nargs="+", default=["s1"])
    parser.add_argument("--test-sessions", nargs="+", default=["s2"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    evaluate(
        checkpoint=args.checkpoint,
        root=args.root,
        train_sessions=args.train_sessions,
        test_sessions=args.test_sessions,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
