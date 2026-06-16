from __future__ import annotations

import torch
import torch.nn.functional as F


def first_order_distance(z: torch.Tensor, loss_type: str = "l2") -> torch.Tensor:
    """D3-style first-order adjacent-frame distance in embedding space.

    Args:
        z: `[B, T, C]` frame embeddings.
        loss_type: `l2` or `cos`, matching the D3 official code options.
    Returns:
        `[B, T-1]` adjacent distance/similarity sequence.
    """
    if z.shape[1] < 2:
        return z.new_zeros((z.shape[0], 1))
    v1, v2 = z[:, :-1, :], z[:, 1:, :]
    if loss_type == "cos":
        return F.cosine_similarity(v1, v2, dim=-1)
    if loss_type == "l2":
        return torch.norm(v1 - v2, p=2, dim=-1)
    raise ValueError(f"Unknown D3 loss_type: {loss_type}")


def second_order_delta(z: torch.Tensor, loss_type: str = "l2") -> torch.Tensor:
    """D3 second-order feature: difference of adjacent first-order distances."""
    d1 = first_order_distance(z, loss_type=loss_type)
    if d1.shape[1] < 2:
        return d1.new_zeros((d1.shape[0], 1))
    return d1[:, 1:] - d1[:, :-1]


def d3_statistics(z: torch.Tensor, loss_type: str = "l2") -> torch.Tensor:
    """D3 statistics used as temporal forensic cues.

    The original D3 evaluation uses the std of the second-order distance as
    the score. For a trainable STF3 branch we keep the original mean/std and
    add robust extrema/energy statistics.
    """
    d1 = first_order_distance(z, loss_type=loss_type)
    d2 = second_order_delta(z, loss_type=loss_type)
    return torch.stack(
        [
            d1.mean(dim=1),
            d1.std(dim=1, unbiased=False),
            d2.mean(dim=1),
            d2.std(dim=1, unbiased=False),
            d2.abs().mean(dim=1),
            d2.abs().amax(dim=1),
        ],
        dim=1,
    )

