from __future__ import annotations

import torch
from torch import nn

from .spatial_branch import SpatialBranch
from .frequency_branch import FrequencyBranch
from .temporal_branch import TemporalBranch


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 2, hidden_dim: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class STF3Detect(nn.Module):
    """STF³-Detect-Lite.

    modes:
      spatial, frequency, temporal, spatial_frequency, spatial_temporal, stf3
    """

    def __init__(
        self,
        mode: str = "stf3",
        spatial_dim: int = 256,
        frequency_dim: int = 128,
        temporal_dim: int = 128,
        pretrained_spatial: bool = False,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        valid = {"spatial", "frequency", "temporal", "spatial_frequency", "spatial_temporal", "stf3"}
        if mode not in valid:
            raise ValueError(f"Unknown mode {mode}; expected one of {sorted(valid)}")
        self.mode = mode

        self.spatial = SpatialBranch(out_dim=spatial_dim, pretrained=pretrained_spatial) if "spatial" in mode or mode == "stf3" else None
        self.frequency = FrequencyBranch(out_dim=frequency_dim) if "frequency" in mode or mode == "stf3" else None
        self.temporal = TemporalBranch(out_dim=temporal_dim) if "temporal" in mode or mode == "stf3" else None

        in_dim = 0
        if self.spatial is not None:
            in_dim += spatial_dim
        if self.frequency is not None:
            in_dim += frequency_dim
        if self.temporal is not None:
            in_dim += temporal_dim
        self.head = MLPHead(in_dim=in_dim, num_classes=num_classes)

    def forward_features(self, frames: torch.Tensor) -> torch.Tensor:
        feats = []
        if self.spatial is not None:
            feats.append(self.spatial(frames))
        if self.frequency is not None:
            feats.append(self.frequency(frames))
        if self.temporal is not None:
            feats.append(self.temporal(frames))
        return torch.cat(feats, dim=1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(frames)
        return self.head(feat)
