from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18, resnet34


def build_cnn(arch: str = "resnet18", num_classes: int = 9,
              in_channels: int = 1, pretrained: bool = False) -> nn.Module:
    """ResNet with a swapped first conv (1-channel) and swapped fc head."""
    weights = None
    if arch == "resnet18":
        model = resnet18(weights=weights)
    elif arch == "resnet34":
        model = resnet34(weights=weights)
    else:
        raise ValueError(f"unknown arch: {arch}")

    if in_channels != 3:
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7,
                                stride=2, padding=3, bias=False)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
