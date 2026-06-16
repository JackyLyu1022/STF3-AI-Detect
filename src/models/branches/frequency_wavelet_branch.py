from __future__ import annotations

import torch
from torch import nn

from src.features.spectral import radial_frequency_stats
from src.features.wavelet import wavelet_energy_stats


class WaveletSpectralBranch(nn.Module):
    """WaveRep/SPAI-inspired frequency branch without the old SmallCNN.

    The branch uses WaveRep's wavelet-band decomposition plus compact radial
    spectral statistics. SPAI's contribution is reflected as spectral-context
    modeling via explicit radial frequency bands; a full SPAI checkpoint can be
    added later as an external score branch.
    """

    def __init__(self, out_dim: int = 256, wavelet_levels: int = 2, fft_bins: int = 6, dropout: float = 0.1) -> None:
        super().__init__()
        self.wavelet_levels = wavelet_levels
        self.fft_bins = fft_bins
        # wavelet: levels * 4 bands * 2 stats * video mean/std
        wave_dim = wavelet_levels * 4 * 2 * 2
        # fft: bins * 2 stats * video mean/std
        fft_dim = fft_bins * 2 * 2
        in_dim = wave_dim + fft_dim
        self.adapter = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim),
        )
        self.out_dim = out_dim

    def raw_features(self, frames: torch.Tensor) -> torch.Tensor:
        # FFT/wavelet statistics are numerically safer in fp32, especially when
        # the training loop uses AMP.
        frames = frames.float()
        wave = wavelet_energy_stats(frames, levels=self.wavelet_levels)
        spec = radial_frequency_stats(frames, bins=self.fft_bins)
        return torch.cat([wave, spec], dim=1)

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raw = self.raw_features(frames)
        return self.adapter(raw), {"frequency_raw": raw}
