from __future__ import annotations

import torch
from torch import nn
from .frequency_branch import SmallCNN


class TemporalBranch(nn.Module):
    """Lightweight temporal branch using adjacent frame differences."""

    def __init__(self, out_dim: int = 128) -> None:
        super().__init__()
        self.encoder = SmallCNN(in_channels=3, out_dim=out_dim)
        self.out_dim = out_dim

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: [B,T,3,H,W]
        if frames.shape[1] < 2:
            diff = torch.zeros_like(frames)
        else:
            diff = torch.abs(frames[:, 1:] - frames[:, :-1])
        b, t, c, h, w = diff.shape
        x = diff.reshape(b * t, c, h, w)
        feat = self.encoder(x)
        return feat.view(b, t, -1).mean(dim=1)
