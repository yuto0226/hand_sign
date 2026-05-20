import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset


class NPZGestureDataset(Dataset):
    """Loads hand gesture NPZ files from a directory tree.

    Expected structure:
        root/
          class_a/  *.npz  (each stores features array of shape (166,))
          class_b/  *.npz
          ...
    """

    def __init__(self, root: str):
        self.classes = sorted(
            d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
        )
        if len(self.classes) < 2:
            raise ValueError(
                f"Need at least 2 classes in {root}, found {len(self.classes)}: {self.classes}"
            )

        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples: list[tuple[str, int]] = []

        for cls in self.classes:
            cls_dir = os.path.join(root, cls)
            for fname in sorted(os.listdir(cls_dir)):
                if fname.endswith(".npz"):
                    self.samples.append(
                        (os.path.join(cls_dir, fname), self.class_to_idx[cls])
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        try:
            data = np.load(path)
            return torch.from_numpy(data["features"]).float(), label

        except Exception as e:
            raise RuntimeError(f"Failed to load features from {path}: {e}") from e


def get_data_loaders(
    root: str,
    batch_size: int,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    dataset = NPZGestureDataset(root)
    n_total = len(dataset)
    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val - n_test
    if n_train < 1 or n_val < 1 or n_test < 1:
        raise ValueError(
            f"Dataset too small ({n_total} samples) for ratios "
            f"val={val_ratio}, test={test_ratio}. "
            f"Got train={n_train}, val={n_val}, test={n_test}."
        )

    indices = torch.randperm(
        n_total, generator=torch.Generator().manual_seed(seed)
    ).tolist()

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    return train_loader, val_loader, test_loader, dataset.classes
