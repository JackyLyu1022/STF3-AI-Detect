from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class SpatialBranch(nn.Module):
    """Frame-level ResNet branch, averaged over time."""

    def __init__(self, out_dim: int = 256, pretrained: bool = False) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        in_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.out_dim = out_dim

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: [B,T,3,H,W] in [0,1]
        b, t, c, h, w = frames.shape
        x = (frames - self.mean) / self.std
        x = x.reshape(b * t, c, h, w)
        feat = self.backbone(x)
        feat = self.proj(feat)
        feat = feat.view(b, t, -1).mean(dim=1)
        return feat
