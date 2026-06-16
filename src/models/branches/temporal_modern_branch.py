from __future__ import annotations

import torch
from torch import nn

from src.features.d3 import d3_statistics
from src.features.geometry import restrav_21d_features


class ModernTemporalBranch(nn.Module):
    """D3 + ReStraV temporal branch.

    - D3: second-order differences of adjacent embedding distances.
    - ReStraV: 21-D representation-trajectory geometry vector.
    """

    def __init__(
        self,
        out_dim: int = 256,
        variant: str = "d3_restrav",
        d3_loss: str = "l2",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        valid = {"d3", "restrav", "d3_restrav"}
        if variant not in valid:
            raise ValueError(f"Unknown temporal variant {variant}; valid={sorted(valid)}")
        self.variant = variant
        self.d3_loss = d3_loss
        in_dim = (6 if "d3" in variant else 0) + (21 if "restrav" in variant else 0)
        self.adapter = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim),
        )
        self.out_dim = out_dim

    def raw_features(self, frame_embeddings: torch.Tensor) -> torch.Tensor:
        feats = []
        if "d3" in self.variant:
            feats.append(d3_statistics(frame_embeddings, loss_type=self.d3_loss))
        if "restrav" in self.variant:
            feats.append(restrav_21d_features(frame_embeddings))
        return torch.cat(feats, dim=1)

    def forward(self, frame_embeddings: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raw = self.raw_features(frame_embeddings)
        return self.adapter(raw), {"temporal_raw": raw}

