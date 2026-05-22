from __future__ import annotations

from typing import cast

import torch.nn as nn
import torchvision.models as models


def build_model(num_classes: int = 12) -> nn.Module:
    backbone = models.efficientnet_b0(weights="IMAGENET1K_V1")
    linear_layer = cast(nn.Linear, backbone.classifier[1])
    in_features = linear_layer.in_features
    backbone.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.SiLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )
    return backbone


def freeze_backbone(model: nn.Module) -> None:
    features = cast(nn.Sequential, getattr(model, "features", None))
    if features is not None:
        for param in features.parameters():
            param.requires_grad = False


def unfreeze_blocks(model: nn.Module, block_indices: list[int]) -> None:
    features = cast(nn.Sequential, getattr(model, "features", None))
    if features is not None:
        for idx in block_indices:
            block = features[idx]
            for param in block.parameters():
                param.requires_grad = True
