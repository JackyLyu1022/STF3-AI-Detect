from __future__ import annotations

import random

import torch
from torch import nn


class TorchDWT(nn.Module):
    """Torch Haar DWT used by WaveRep-style augmentation/features.

    The official WaveRep implementation provides general PyWavelets-backed DWT.
    For this project we keep the same band-decomposition and band-replacement
    idea with a compact differentiable Haar implementation to avoid extra
    runtime dependencies and make it easy to run inside the current project.
    """

    def __init__(self) -> None:
        super().__init__()
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]], dtype=torch.float32)
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]], dtype=torch.float32)
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32)
        filt = torch.stack([ll, lh, hl, hh], dim=0).view(4, 1, 2, 2)
        self.register_buffer("filters", filt, persistent=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [N,C,H,W]
        n, c, h, w = x.shape
        if h % 2 == 1:
            x = x[..., :-1, :]
        if w % 2 == 1:
            x = x[..., :-1]
        weight = self.filters.repeat(c, 1, 1, 1)
        y = torch.nn.functional.conv2d(x, weight, stride=2, groups=c)
        y = y.view(n, c, 4, y.shape[-2], y.shape[-1])
        return y[:, :, 0], y[:, :, 1], y[:, :, 2], y[:, :, 3]


def haar_idwt(ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor, out_hw: tuple[int, int] | None = None) -> torch.Tensor:
    """Inverse of `TorchDWT` for Haar bands."""
    n, c, h, w = ll.shape
    y = ll.new_zeros((n, c, h * 2, w * 2))
    y[..., 0::2, 0::2] = (ll - lh - hl + hh) * 0.5
    y[..., 0::2, 1::2] = (ll - lh + hl - hh) * 0.5
    y[..., 1::2, 0::2] = (ll + lh - hl - hh) * 0.5
    y[..., 1::2, 1::2] = (ll + lh + hl + hh) * 0.5
    if out_hw is not None:
        y = y[..., : out_hw[0], : out_hw[1]]
    return y


class WaveRepAugment(nn.Module):
    """WaveRep-style forensic-oriented wavelet band replacement.

    WaveRep replaces selected wavelet sub-bands between real/fake images to
    push the detector toward robust forensic traces. ``batch`` mode uses a
    shuffled sample from the current batch. ``bank`` mode keeps the previous
    training sample as a detached donor, so augmentation also works when the
    training batch size is one.
    """

    def __init__(self, prob: float = 0.1, mode: str = "batch") -> None:
        super().__init__()
        if mode not in {"batch", "bank"}:
            raise ValueError(f"Unknown WaveRep augmentation mode: {mode}")
        self.prob = prob
        self.mode = mode
        self.dwt = TorchDWT()
        self._donor_bank: torch.Tensor | None = None

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if not self.training or self.prob <= 0:
            return frames

        current = frames.detach()
        if self.mode == "bank":
            donor_frames = self._donor_bank
            self._donor_bank = current[-1:].clone()
            if donor_frames is None or donor_frames.shape[1:] != frames.shape[1:]:
                return frames
            donor_frames = donor_frames.to(device=frames.device, dtype=frames.dtype)
            donor_frames = donor_frames.expand(frames.shape[0], -1, -1, -1, -1)
        else:
            if frames.shape[0] < 2:
                return frames
            perm_b = torch.randperm(frames.shape[0], device=frames.device)
            donor_frames = frames[perm_b]

        if random.random() >= self.prob:
            return frames
        b, t, c, h, w = frames.shape
        x = frames.reshape(b * t, c, h, w)
        donor = donor_frames.reshape(b * t, c, h, w)
        ll_x, lh_x, hl_x, hh_x = self.dwt(x)
        ll_d, lh_d, hl_d, hh_d = self.dwt(donor)

        # Once augmentation is triggered, replace either the low-frequency
        # band or all detail bands for every frame. This keeps ``prob`` as the
        # actual sample-level augmentation probability instead of applying it
        # twice and making the effective replacement rate approximately p^2.
        use_low = torch.rand((b * t, 1, 1, 1), device=frames.device) < 0.5
        use_high = ~use_low
        ll = torch.where(use_low, ll_d, ll_x)
        lh = torch.where(use_high, lh_d, lh_x)
        hl = torch.where(use_high, hl_d, hl_x)
        hh = torch.where(use_high, hh_d, hh_x)
        y = haar_idwt(ll, lh, hl, hh, out_hw=(h, w))
        return y.clamp(0, 1).view(b, t, c, h, w)


def wavelet_energy_stats(frames: torch.Tensor, levels: int = 2) -> torch.Tensor:
    """Multi-level Haar wavelet energy statistics per video."""
    dwt = TorchDWT().to(device=frames.device, dtype=frames.dtype)
    b, t, c, h, w = frames.shape
    x = frames.reshape(b * t, c, h, w)
    feats = []
    cur = x
    for _ in range(levels):
        ll, lh, hl, hh = dwt(cur)
        bands = [ll, lh, hl, hh]
        for band in bands:
            energy = band.square().mean(dim=(1, 2, 3))
            abs_mean = band.abs().mean(dim=(1, 2, 3))
            feats.extend([energy, abs_mean])
        cur = ll
    f = torch.stack(feats, dim=1).view(b, t, -1)
    return torch.cat([f.mean(dim=1), f.std(dim=1, unbiased=False)], dim=1)
