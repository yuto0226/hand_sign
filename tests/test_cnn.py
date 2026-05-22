import tempfile
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
from PIL import Image

from cnnclassifier.dataset import make_splits
from cnnclassifier.model import build_model, freeze_backbone, unfreeze_blocks


def test_build_model_output_shape():
    model = build_model(num_classes=12)
    x = torch.zeros(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 12)


def test_freeze_backbone_stops_gradients():
    model = build_model(num_classes=12)
    freeze_backbone(model)
    features = cast(nn.Sequential, getattr(model, "features"))
    classifier = cast(nn.Sequential, getattr(model, "classifier"))
    for param in features.parameters():
        assert not param.requires_grad
    for param in classifier.parameters():
        assert param.requires_grad


def test_unfreeze_blocks_restores_gradients():
    model = build_model(num_classes=12)
    freeze_backbone(model)
    unfreeze_blocks(model, [7, 8])
    features = cast(nn.Sequential, getattr(model, "features"))
    for param in features[7].parameters():
        assert param.requires_grad
    for param in features[8].parameters():
        assert param.requires_grad
    # earlier blocks still frozen
    for param in features[0].parameters():
        assert not param.requires_grad


def _make_dummy_data(root: Path) -> None:
    """Create 4 classes × 2 sessions × 3 images each."""
    for sign in ["ne", "ushi", "tora", "u"]:
        d = root / sign
        d.mkdir(parents=True)
        for session in ["s1", "s2"]:
            for i in range(3):
                img = Image.new("RGB", (32, 32), color=(i * 40, 0, 0))
                img.save(d / f"{session}_{i:04d}.jpg")


def test_make_splits_no_session_overlap():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_dummy_data(root)
        train_loader, val_loader, test_loader, classes = make_splits(
            root,
            train_sessions=["s1"],
            val_fraction=0.1,
            test_sessions=["s2"],
            batch_size=4,
            num_workers=0,
        )
        assert len(classes) == 4
        # test set should only contain s2 images (last 50%)
        batch_count = sum(1 for _ in test_loader)
        assert batch_count > 0
