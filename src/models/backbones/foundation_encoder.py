from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class FoundationSpec:
    name: str
    embed_dim: int
    image_size: int = 224
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)


FOUNDATION_SPECS: dict[str, FoundationSpec] = {
    # ReStraV official implementation uses torch.hub DINOv2 ViT-S/14.
    "dinov2_vits14": FoundationSpec("dinov2_vits14", embed_dim=384),
    "dinov2_vitb14": FoundationSpec("dinov2_vitb14", embed_dim=768),
    # D3 official implementation supports CLIP / XCLIP / DINO family encoders.
    "clip-vit-base-patch16": FoundationSpec(
        "clip-vit-base-patch16",
        embed_dim=768,
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    ),
    "xclip-base-patch16": FoundationSpec(
        "xclip-base-patch16",
        embed_dim=768,
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    ),
    "hf-dinov2-base": FoundationSpec("hf-dinov2-base", embed_dim=768),
}


class FoundationVisionEncoder(nn.Module):
    """Frozen DINOv2 / CLIP / XCLIP encoder used by STF3-New.

    This module intentionally follows the encoders used in the 2025 papers:
    - ReStraV: DINOv2 ViT-S/14 via torch.hub and `forward_features`.
    - D3: CLIPVisionModel / XCLIPVisionModel / DINO through HuggingFace.

    Input frames are expected as `[B, T, 3, H, W]` in `[0, 1]`.
    Output is a per-frame embedding tensor `[B, T, C]`.
    """

    def __init__(
        self,
        backbone: str = "dinov2_vits14",
        freeze: bool = True,
        image_size: int = 224,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        if backbone not in FOUNDATION_SPECS:
            raise ValueError(f"Unknown foundation backbone: {backbone}. Valid: {sorted(FOUNDATION_SPECS)}")
        self.backbone_name = backbone
        self.spec = FOUNDATION_SPECS[backbone]
        self.embed_dim = self.spec.embed_dim
        self.image_size = image_size
        self.local_files_only = local_files_only

        mean = torch.tensor(self.spec.mean, dtype=torch.float32).view(1, 1, 3, 1, 1)
        std = torch.tensor(self.spec.std, dtype=torch.float32).view(1, 1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

        self.backend = "torchhub_dinov2" if backbone.startswith("dinov2_") else "hf"
        self.model = self._load_model(backbone)
        if freeze:
            self.freeze()

    def _load_model(self, backbone: str) -> nn.Module:
        if backbone.startswith("dinov2_"):
            # Same loading style as ReStraV's official `dinov2_features.py`.
            cache_dir = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
            if cache_dir.exists():
                return torch.hub.load(str(cache_dir), backbone, source="local").eval()
            return torch.hub.load("facebookresearch/dinov2", backbone, trust_repo=True).eval()

        from transformers import AutoModel, CLIPVisionModel, XCLIPVisionModel

        if backbone == "clip-vit-base-patch16":
            return CLIPVisionModel.from_pretrained(
                "openai/clip-vit-base-patch16",
                local_files_only=self.local_files_only,
            ).eval()
        if backbone == "xclip-base-patch16":
            return XCLIPVisionModel.from_pretrained(
                "microsoft/xclip-base-patch16",
                local_files_only=self.local_files_only,
            ).eval()
        if backbone == "hf-dinov2-base":
            return AutoModel.from_pretrained(
                "facebook/dinov2-base",
                local_files_only=self.local_files_only,
            ).eval()
        raise ValueError(backbone)

    def freeze(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = False

    def preprocess(self, frames: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = frames.shape
        x = frames
        if h != self.image_size or w != self.image_size:
            x = x.reshape(b * t, c, h, w)
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bicubic", align_corners=False)
            x = x.view(b, t, c, self.image_size, self.image_size)
        return (x - self.mean) / self.std

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = frames.shape
        x = self.preprocess(frames).reshape(b * t, c, self.image_size, self.image_size)
        if self.backend == "torchhub_dinov2":
            feats = self.model.forward_features(x)
            if isinstance(feats, dict) and "x_norm_clstoken" in feats:
                z = feats["x_norm_clstoken"]
            else:
                z = self.model(x)
        else:
            out = self.model(x, output_hidden_states=True)
            z = getattr(out, "pooler_output", None)
            if z is None:
                z = out.last_hidden_state[:, 0]
        return z.reshape(b, t, -1)
