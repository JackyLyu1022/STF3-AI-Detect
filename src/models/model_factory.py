from __future__ import annotations

from src.models.stf3_detect import STF3Detect
from src.models.stf3_modern import MODERN_MODES, STF3Modern


LEGACY_MODES = {"spatial", "frequency", "temporal", "spatial_frequency", "spatial_temporal", "stf3"}
ALL_MODEL_MODES = sorted(LEGACY_MODES | MODERN_MODES)


def build_model(
    mode: str,
    pretrained_spatial: bool = False,
    foundation_backbone: str = "dinov2_vits14",
    freeze_foundation: bool = True,
    local_files_only: bool = False,
    image_size: int = 224,
    wavelet_aug_prob: float = 0.1,
    wavelet_aug_mode: str = "batch",
    branch_dropout: float = 0.1,
    d3_loss: str = "l2",
):
    if mode in LEGACY_MODES:
        return STF3Detect(mode=mode, pretrained_spatial=pretrained_spatial)
    if mode in MODERN_MODES:
        return STF3Modern(
            mode=mode,
            foundation_backbone=foundation_backbone,
            freeze_foundation=freeze_foundation,
            local_files_only=local_files_only,
            image_size=image_size,
            wavelet_aug_prob=wavelet_aug_prob,
            wavelet_aug_mode=wavelet_aug_mode,
            branch_dropout=branch_dropout,
            d3_loss=d3_loss,
        )
    raise ValueError(f"Unknown model mode {mode}; valid={ALL_MODEL_MODES}")
