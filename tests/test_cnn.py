from typing import cast

import torch
import torch.nn as nn

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
