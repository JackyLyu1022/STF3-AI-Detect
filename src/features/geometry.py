from __future__ import annotations

import torch
import torch.nn.functional as F


def temporal_geometry(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """ReStraV-style step distances and turning angles.

    ReStraV treats per-frame DINOv2 embeddings as a trajectory and computes
    distances plus curvature/turning-angle cues.
    """
    if z.shape[1] < 2:
        d = z.new_zeros((z.shape[0], 1))
        theta = z.new_zeros((z.shape[0], 1))
        return d, theta
    delta = z[:, 1:, :] - z[:, :-1, :]
    d = delta.norm(dim=-1)
    if delta.shape[1] < 2:
        theta = z.new_zeros((z.shape[0], 1))
    else:
        cos = F.cosine_similarity(delta[:, :-1, :], delta[:, 1:, :], dim=-1)
        theta = torch.rad2deg(torch.acos(cos.clamp(-1.0, 1.0)))
    return d, theta


def _pad_or_trim(x: torch.Tensor, n: int) -> torch.Tensor:
    if x.shape[1] >= n:
        return x[:, :n]
    pad = x.new_zeros((x.shape[0], n - x.shape[1]))
    return torch.cat([x, pad], dim=1)


def _moment4(x: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            x.mean(dim=1),
            x.amin(dim=1),
            x.amax(dim=1),
            x.var(dim=1, unbiased=False),
        ],
        dim=1,
    )


def restrav_21d_features(z: torch.Tensor) -> torch.Tensor:
    """ReStraV official 21-D feature layout.

    - 7 early stepwise distances
    - 6 early turning angles
    - 8 summary statistics for distances and angles
    """
    d, theta = temporal_geometry(z)
    d7 = _pad_or_trim(d, 7)
    t6 = _pad_or_trim(theta, 6)
    stats = torch.cat([_moment4(d), _moment4(theta)], dim=1)
    return torch.cat([d7, t6, stats], dim=1)

