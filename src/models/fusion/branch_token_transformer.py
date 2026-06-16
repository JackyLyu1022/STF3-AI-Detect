from __future__ import annotations

import torch
from torch import nn


class BranchTokenFusionTransformer(nn.Module):
    """Fuse S/T/F branch tokens with a lightweight Transformer encoder."""

    def __init__(
        self,
        token_dim: int = 256,
        num_classes: int = 2,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_branches: int = 3,
        branch_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.branch_dropout = branch_dropout
        self.cls = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.branch_embed = nn.Parameter(torch.randn(1, max_branches + 1, token_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(token_dim)
        self.head = nn.Linear(token_dim, num_classes)

    def _apply_branch_dropout(self, tokens: torch.Tensor) -> torch.Tensor:
        if not self.training or self.branch_dropout <= 0 or tokens.shape[1] <= 1:
            return tokens
        keep = (torch.rand(tokens.shape[:2], device=tokens.device) > self.branch_dropout).float()
        # Ensure every sample keeps at least one branch.
        empty = keep.sum(dim=1) == 0
        if empty.any():
            keep[empty, torch.randint(tokens.shape[1], (int(empty.sum()),), device=tokens.device)] = 1.0
        return tokens * keep.unsqueeze(-1)

    def forward(self, branch_tokens: list[torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not branch_tokens:
            raise ValueError("At least one branch token is required.")
        x = torch.stack(branch_tokens, dim=1)  # [B,N,D]
        x = self._apply_branch_dropout(x)
        b, n, _ = x.shape
        cls = self.cls.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.branch_embed[:, : n + 1]
        enc = self.encoder(x)
        cls_out = self.norm(enc[:, 0])
        logits = self.head(cls_out)
        # Approximate branch importance by CLS-token dot products.
        branch_scores = torch.softmax((enc[:, 1:] * cls_out.unsqueeze(1)).sum(dim=-1), dim=1)
        return logits, {"branch_weights": branch_scores, "fusion_tokens": enc}


class GatedConcatFusion(nn.Module):
    """Simpler ablation fusion: gated weighted tokens then classifier."""

    def __init__(self, token_dim: int = 256, num_classes: int = 2, max_branches: int = 3) -> None:
        super().__init__()
        self.gate = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, 1))
        self.head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, num_classes),
        )

    def forward(self, branch_tokens: list[torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = torch.stack(branch_tokens, dim=1)
        w = torch.softmax(self.gate(x).squeeze(-1), dim=1)
        fused = torch.sum(x * w.unsqueeze(-1), dim=1)
        return self.head(fused), {"branch_weights": w}

