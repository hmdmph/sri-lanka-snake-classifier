from __future__ import annotations

import torch.nn as nn
from torchvision import models

_BUILDERS = {
    "mobilenet_v3_small": (
        models.mobilenet_v3_small,
        models.MobileNet_V3_Small_Weights.DEFAULT,
    ),
    "mobilenet_v3_large": (
        models.mobilenet_v3_large,
        models.MobileNet_V3_Large_Weights.DEFAULT,
    ),
}


def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if model_name not in _BUILDERS:
        raise ValueError(f"Unknown model_name {model_name!r}, choose from {list(_BUILDERS)}")
    builder, weights = _BUILDERS[model_name]
    model = builder(weights=weights if pretrained else None)

    # torchvision's MobileNetV3 classifier is Sequential(Linear, Hardswish,
    # Dropout, Linear) -- swap only the final Linear so the learned feature
    # projection (and its ImageNet init) is kept.
    last_layer = model.classifier[-1]
    model.classifier[-1] = nn.Linear(last_layer.in_features, num_classes)
    return model


def backbone_parameters(model: nn.Module):
    return model.features.parameters()


def head_parameters(model: nn.Module):
    return model.classifier.parameters()


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for p in backbone_parameters(model):
        p.requires_grad = trainable
