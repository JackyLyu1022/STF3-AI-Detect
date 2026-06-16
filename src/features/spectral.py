from __future__ import annotations

import torch


def rgb_to_gray(frames: torch.Tensor) -> torch.Tensor:
    return 0.299 * frames[:, :, 0] + 0.587 * frames[:, :, 1] + 0.114 * frames[:, :, 2]


def fft_log_magnitude(frames: torch.Tensor) -> torch.Tensor:
    gray = rgb_to_gray(frames)
    fft = torch.fft.fft2(gray, norm="ortho")
    mag = torch.log1p(torch.abs(fft))
    return torch.fft.fftshift(mag, dim=(-2, -1))


def radial_frequency_stats(frames: torch.Tensor, bins: int = 6) -> torch.Tensor:
    """Compact spectral statistics for video frames.

    Returns per-video stats over radial frequency bands. This replaces the old
    `FFT -> SmallCNN` path with explicit spectral cues that can be fused with
    wavelet features and optional SPAI scores.
    """
    mag = fft_log_magnitude(frames)  # [B,T,H,W]
    b, t, h, w = mag.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=mag.device, dtype=mag.dtype),
        torch.linspace(-1.0, 1.0, w, device=mag.device, dtype=mag.dtype),
        indexing="ij",
    )
    r = torch.sqrt(xx.square() + yy.square()).clamp_max(1.0)
    edges = torch.linspace(0.0, 1.0, bins + 1, device=mag.device, dtype=mag.dtype)
    feats = []
    for i in range(bins):
        mask = (r >= edges[i]) & (r < edges[i + 1] if i < bins - 1 else r <= edges[i + 1])
        vals = mag[..., mask]
        feats.append(vals.mean(dim=-1))
        feats.append(vals.std(dim=-1, unbiased=False))
    stat = torch.stack(feats, dim=-1)  # [B,T,2*bins]
    return torch.cat([stat.mean(dim=1), stat.std(dim=1, unbiased=False)], dim=1)

