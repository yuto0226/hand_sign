from __future__ import annotations

import argparse
from typing import NamedTuple

import torch
from tqdm import tqdm

from dataset import get_data_loaders
from model import HandSignTransformer


def load_checkpoint(path: str, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"]
    metadata = {
        "classes": checkpoint.get("classes", []),
        "d_model": checkpoint.get("d_model", 64),
        "nhead": checkpoint.get("nhead", 4),
        "num_layers": checkpoint.get("num_layers", 2),
        "dim_feedforward": checkpoint.get("dim_feedforward", 128),
    }
    return state_dict, metadata


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    preds_list, labels_list = [], []
    for inputs, labels in tqdm(loader, desc="  infer", leave=False):
        preds_list.append(model(inputs.to(device)).argmax(1).cpu())
        labels_list.append(
            labels if isinstance(labels, torch.Tensor) else torch.tensor(labels)
        )
    return torch.cat(preds_list), torch.cat(labels_list)


def build_confusion_matrix(
    preds: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """cm[true][pred] — row = ground truth, col = predicted"""
    return torch.bincount(
        labels * num_classes + preds, minlength=num_classes**2
    ).reshape(num_classes, num_classes)


class Metrics(NamedTuple):
    recall: torch.Tensor
    precision: torch.Tensor
    f_measure: torch.Tensor
    accuracy: torch.Tensor
    avg_precision: torch.Tensor
    avg_recall: torch.Tensor
    avg_f_measure: torch.Tensor


def compute_metrics(cm: torch.Tensor) -> Metrics:
    cm_f = cm.float()
    tp = cm_f.diag()
    row_sum = cm_f.sum(dim=1)
    col_sum = cm_f.sum(dim=0)
    recall = torch.where(row_sum > 0, tp / row_sum, torch.zeros_like(tp))
    precision = torch.where(col_sum > 0, tp / col_sum, torch.zeros_like(tp))
    denom = precision + recall
    f_measure = torch.where(
        denom > 0, 2 * precision * recall / denom, torch.zeros_like(tp)
    )
    accuracy = tp.sum() / cm_f.sum()
    return Metrics(
        recall=recall,
        precision=precision,
        f_measure=f_measure,
        accuracy=accuracy,
        avg_precision=precision.mean(),
        avg_recall=recall.mean(),
        avg_f_measure=f_measure.mean(),
    )


def print_report(
    cm: torch.Tensor,
    metrics: Metrics,
    class_names: list[str],
    split_name: str = "Val",
) -> None:
    nc = len(class_names)
    w = max(len(n) for n in class_names) + 1

    print(f"\n  {split_name}  —  Confusion Matrix")
    print(f"{'─' * 72}")
    abbr = [n[:4] for n in class_names]
    print(f"{'':>{w}}" + "".join(f"{a:>7}" for a in abbr))
    print("-" * (w + nc * 7))
    for i, name in enumerate(class_names):
        row = f"{name:>{w}}"
        for j in range(nc):
            val = cm[i, j].item()
            cell = f"{val}*" if i == j else str(val)
            row += f"{cell:>7}"
        print(row)

    print("\n  Per-class metrics")
    print(f"{'─' * 72}")
    print(f"{'Class':>{w}}  {'Recall':>10}  {'Precision':>10}  {'F-measure':>10}")
    print("-" * (w + 38))
    for n, name in enumerate(class_names):
        print(
            f"{name:>{w}}  {metrics.recall[n].item():>10.4f}  "
            f"{metrics.precision[n].item():>10.4f}  {metrics.f_measure[n].item():>10.4f}"
        )

    print("\n  Overall")
    print(f"{'─' * 72}")
    print(
        f"  Accuracy      : {metrics.accuracy.item():.4f}  ({metrics.accuracy.item() * 100:.2f}%)"
    )
    print(f"  Avg Precision : {metrics.avg_precision.item():.4f}")
    print(f"  Avg Recall    : {metrics.avg_recall.item():.4f}")
    print(f"  Avg F-measure : {metrics.avg_f_measure.item():.4f}")
    print(f"{'─' * 72}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate HandSignTransformer")
    parser.add_argument("--data", type=str, default="data/npz")
    parser.add_argument("--checkpoint", type=str, default="best.pth")
    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    print(f"[*] Loading checkpoint: {args.checkpoint}")
    state_dict, ckpt_meta = load_checkpoint(args.checkpoint, device)
    classes = ckpt_meta["classes"]

    print(f"[*] Loading dataset from {args.data} ...")
    _, _, test_loader, data_classes = get_data_loaders(
        root=args.data,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    if classes:
        if classes != data_classes:
            raise ValueError(
                f"Checkpoint classes {classes} do not match dataset classes {data_classes}."
            )
    else:
        classes = data_classes
    print(f"[*] Classes: {classes}  test: {len(test_loader.dataset)}")  # type: ignore[arg-type]

    model = HandSignTransformer(
        num_classes=len(classes),
        d_model=ckpt_meta["d_model"],
        nhead=ckpt_meta["nhead"],
        num_layers=ckpt_meta["num_layers"],
        dim_feedforward=ckpt_meta["dim_feedforward"],
    ).to(device)
    model.load_state_dict(state_dict)

    preds, labels = collect_predictions(model, test_loader, device)
    cm = build_confusion_matrix(preds, labels, num_classes=len(classes))
    metrics = compute_metrics(cm)
    print_report(cm, metrics, class_names=classes, split_name="Test")


if __name__ == "__main__":
    main()
