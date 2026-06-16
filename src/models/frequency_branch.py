from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    def __init__(self, in_channels: int = 1, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(128, out_dim), nn.ReLU(inplace=True), nn.Dropout(0.2))
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.net(x))


class FrequencyBranch(nn.Module):
    """FFT log-magnitude branch."""

    def __init__(self, out_dim: int = 128) -> None:
        super().__init__()
        self.encoder = SmallCNN(in_channels=1, out_dim=out_dim)
        self.out_dim = out_dim

    @staticmethod
    def frames_to_spectrum(frames: torch.Tensor) -> torch.Tensor:
        # frames: [B,T,3,H,W] in [0,1]
        gray = 0.299 * frames[:, :, 0] + 0.587 * frames[:, :, 1] + 0.114 * frames[:, :, 2]
        fft = torch.fft.fft2(gray, norm="ortho")
        mag = torch.log1p(torch.abs(fft))
        mag = torch.fft.fftshift(mag, dim=(-2, -1))
        # per-frame standardization
        mean = mag.mean(dim=(-2, -1), keepdim=True)
        std = mag.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        mag = (mag - mean) / std
        return mag.unsqueeze(2)  # [B,T,1,H,W]

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        spec = self.frames_to_spectrum(frames)
        b, t, c, h, w = spec.shape
        x = spec.reshape(b * t, c, h, w)
        feat = self.encoder(x)
        return feat.view(b, t, -1).mean(dim=1)
