from __future__ import annotations

import torch
from torch import nn

from src.features.wavelet import WaveRepAugment
from src.models.backbones.foundation_encoder import FoundationVisionEncoder
from src.models.branches.frequency_wavelet_branch import WaveletSpectralBranch
from src.models.branches.spatial_foundation_branch import FoundationSpatialBranch
from src.models.branches.temporal_modern_branch import ModernTemporalBranch
from src.models.fusion.branch_token_transformer import BranchTokenFusionTransformer, GatedConcatFusion


MODERN_MODES = {
    "spatial_dino",
    "temporal_d3",
    "temporal_restrav",
    "temporal_d3_restrav",
    "frequency_wave",
    "spatial_frequency_new",
    "spatial_temporal_new",
    "stf3_new",
    "stf3_new_concat",
}


class STF3Modern(nn.Module):
    """Paper-driven modern STF3 implementation.

    It keeps the STF3 idea but replaces old components with methods used by
    2025 papers:
      - ReStraV/D3 foundation encoders (DINOv2/XCLIP/CLIP)
      - D3 second-order temporal dynamics
      - ReStraV trajectory geometry
      - WaveRep wavelet-band frequency modeling/augmentation
      - Branch-token transformer fusion
    """

    def __init__(
        self,
        mode: str = "stf3_new",
        foundation_backbone: str = "dinov2_vits14",
        token_dim: int = 256,
        num_classes: int = 2,
        freeze_foundation: bool = True,
        local_files_only: bool = False,
        image_size: int = 224,
        d3_loss: str = "l2",
        wavelet_aug_prob: float = 0.1,
        wavelet_aug_mode: str = "batch",
        branch_dropout: float = 0.1,
        fusion: str | None = None,
    ) -> None:
        super().__init__()
        if mode not in MODERN_MODES:
            raise ValueError(f"Unknown STF3Modern mode {mode}; valid={sorted(MODERN_MODES)}")
        self.mode = mode
        self.branch_names: list[str] = []
        self.wavelet_aug = WaveRepAugment(prob=wavelet_aug_prob, mode=wavelet_aug_mode)

        needs_foundation = mode != "frequency_wave"
        self.foundation = (
            FoundationVisionEncoder(
                foundation_backbone,
                freeze=freeze_foundation,
                image_size=image_size,
                local_files_only=local_files_only,
            )
            if needs_foundation
            else None
        )
        embed_dim = self.foundation.embed_dim if self.foundation is not None else token_dim

        self.spatial = None
        self.temporal = None
        self.frequency = None

        if mode in {"spatial_dino", "spatial_frequency_new", "spatial_temporal_new", "stf3_new", "stf3_new_concat"}:
            self.spatial = FoundationSpatialBranch(embed_dim, out_dim=token_dim)
            self.branch_names.append("spatial")

        if mode in {
            "temporal_d3",
            "temporal_restrav",
            "temporal_d3_restrav",
            "spatial_temporal_new",
            "stf3_new",
            "stf3_new_concat",
        }:
            if mode == "temporal_d3":
                variant = "d3"
            elif mode == "temporal_restrav":
                variant = "restrav"
            else:
                variant = "d3_restrav"
            self.temporal = ModernTemporalBranch(out_dim=token_dim, variant=variant, d3_loss=d3_loss)
            self.branch_names.append("temporal")

        if mode in {"frequency_wave", "spatial_frequency_new", "stf3_new", "stf3_new_concat"}:
            self.frequency = WaveletSpectralBranch(out_dim=token_dim)
            self.branch_names.append("frequency")

        fusion = fusion or ("gated" if mode.endswith("_concat") else "transformer")
        if fusion == "transformer":
            self.fusion = BranchTokenFusionTransformer(
                token_dim=token_dim,
                num_classes=num_classes,
                branch_dropout=branch_dropout,
                max_branches=max(3, len(self.branch_names)),
            )
        elif fusion == "gated":
            self.fusion = GatedConcatFusion(token_dim=token_dim, num_classes=num_classes, max_branches=max(3, len(self.branch_names)))
        else:
            raise ValueError(f"Unknown fusion: {fusion}")

        self.aux_heads = nn.ModuleDict({name: nn.Linear(token_dim, num_classes) for name in self.branch_names})

    def forward(self, frames: torch.Tensor, return_dict: bool = False):
        if self.training and self.frequency is not None:
            frames = self.wavelet_aug(frames)

        z = self.foundation(frames) if self.foundation is not None else None
        tokens: list[torch.Tensor] = []
        aux_logits: dict[str, torch.Tensor] = {}
        details: dict[str, torch.Tensor] = {}

        if self.spatial is not None:
            assert z is not None
            tok, det = self.spatial(z)
            tokens.append(tok)
            aux_logits["spatial"] = self.aux_heads["spatial"](tok)
            details.update(det)

        if self.temporal is not None:
            assert z is not None
            tok, det = self.temporal(z)
            tokens.append(tok)
            aux_logits["temporal"] = self.aux_heads["temporal"](tok)
            details.update(det)

        if self.frequency is not None:
            tok, det = self.frequency(frames)
            tokens.append(tok)
            aux_logits["frequency"] = self.aux_heads["frequency"](tok)
            details.update(det)

        logits, fusion_details = self.fusion(tokens)
        details.update(fusion_details)
        if return_dict:
            return {
                "logits": logits,
                "aux_logits": aux_logits,
                "details": details,
                "branch_names": self.branch_names,
            }
        return logits
