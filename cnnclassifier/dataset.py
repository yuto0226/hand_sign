from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.TrivialAugmentWide(),
        transforms.RandomRotation(15),
        transforms.RandomPerspective(distortion_scale=0.2),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
    ]
)

VAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ]
)


def _session(path: str) -> str:
    """'data/cnn/ne/s1_0042.jpg' → 's1'"""
    return Path(path).stem.split("_")[0]


def make_splits(
    root: str | Path = "data/cnn",
    train_sessions: Sequence[str] = ("s1",),
    val_fraction: float = 0.1,
    test_sessions: Sequence[str] = ("s2",),
    batch_size: int = 32,
    num_workers: int = 2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Return (train_loader, val_loader, test_loader, class_names).

    Session-based split: no random shuffle across sessions to prevent
    leakage from near-identical consecutive frames.
    """
    train_set_full = datasets.ImageFolder(root, transform=TRAIN_TRANSFORM)
    eval_set_full = datasets.ImageFolder(root, transform=VAL_TRANSFORM)
    classes = train_set_full.classes

    train_sessions_set = set(train_sessions)
    test_sessions_set = set(test_sessions)

    train_idx: list[int] = []
    test_idx_by_class: dict[int, list[int]] = {}

    for i, (path, cls) in enumerate(train_set_full.imgs):
        s = _session(path)
        if s in train_sessions_set:
            train_idx.append(i)
        elif s in test_sessions_set:
            test_idx_by_class.setdefault(cls, []).append(i)

    # fallback: no session-tagged files → random split by class
    if not train_idx:
        import random as _random

        all_by_class: dict[int, list[int]] = {}
        for i, (_, cls) in enumerate(train_set_full.imgs):
            all_by_class.setdefault(cls, []).append(i)
        _random.seed(seed)
        for indices in all_by_class.values():
            _random.shuffle(indices)
            cut = max(1, int(len(indices) * 0.8))
            train_idx.extend(indices[:cut])
            test_idx_by_class.setdefault(0, []).extend(indices[cut:])
        # rebuild test_idx_by_class properly for the test split below
        test_idx_by_class = {}
        for i, (_, cls) in enumerate(train_set_full.imgs):
            if i not in set(train_idx):
                test_idx_by_class.setdefault(cls, []).append(i)

    # carve val evenly from train (every Nth sample)
    val_count = max(1, int(len(train_idx) * val_fraction))
    step = max(1, len(train_idx) // val_count)
    val_idx = train_idx[::step][:val_count]
    val_set_idx = set(val_idx)
    train_idx = [i for i in train_idx if i not in val_set_idx]

    # test: only last 50% of each class's test-session samples
    test_idx: list[int] = []
    for indices in test_idx_by_class.values():
        n = len(indices)
        test_idx.extend(indices[n // 2 :])

    def _loader(dataset, indices, shuffle: bool) -> DataLoader:
        return DataLoader(
            Subset(dataset, indices),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    return (
        _loader(train_set_full, train_idx, shuffle=True),
        _loader(eval_set_full, val_idx, shuffle=False),
        _loader(eval_set_full, test_idx, shuffle=False),
        classes,
    )
