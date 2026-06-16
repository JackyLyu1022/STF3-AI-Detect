from __future__ import annotations

import torch
from torch import nn


class TemporalAttentionPool(nn.Module):
    """Attention pooling over frame embeddings instead of plain mean pooling."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, max(64, in_dim // 4)),
            nn.GELU(),
            nn.Linear(max(64, in_dim // 4), 1),
        )

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.score(z).squeeze(-1), dim=1)
        pooled = torch.sum(z * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class FoundationSpatialBranch(nn.Module):
    """Modern spatial branch using frozen foundation frame embeddings."""

    def __init__(self, in_dim: int, out_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.pool = TemporalAttentionPool(in_dim)
        self.adapter = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, frame_embeddings: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pooled, attn = self.pool(frame_embeddings)
        return self.adapter(pooled), {"spatial_frame_attn": attn}

