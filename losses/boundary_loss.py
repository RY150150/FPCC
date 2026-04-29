"""Simple boundary-related loss functions for BA-FPCC-OBB."""

import torch
import torch.nn.functional as F


def boundary_weighted_loss(point_loss, boundary_labels, lambda_boundary=1.0):
    """Weighted mean point loss.

    point_loss: [B,N] or [N]
    boundary_labels: same shape, {0,1}
    """
    if point_loss.shape != boundary_labels.shape:
        raise ValueError(f"shape mismatch: point_loss {point_loss.shape} vs boundary_labels {boundary_labels.shape}")
    weights = 1.0 + lambda_boundary * boundary_labels.float()
    return (point_loss * weights).mean()


def boundary_bce_loss(boundary_logits, boundary_labels):
    """BCEWithLogits loss for boundary branch.

    boundary_logits: [B,N] or [N]
    boundary_labels: same shape
    """
    if boundary_logits.shape != boundary_labels.shape:
        raise ValueError(f"shape mismatch: boundary_logits {boundary_logits.shape} vs boundary_labels {boundary_labels.shape}")
    return F.binary_cross_entropy_with_logits(boundary_logits, boundary_labels.float())


if __name__ == "__main__":
    pl = torch.rand(2, 16)
    bl = torch.randint(0, 2, (2, 16))
    lw = boundary_weighted_loss(pl, bl, lambda_boundary=1.0)
    lw0 = boundary_weighted_loss(pl, bl, lambda_boundary=0.0)
    print("weighted loss:", lw.item())
    print("plain mean loss:", lw0.item(), "raw mean:", pl.mean().item())

    logits = torch.randn(2, 16)
    bce = boundary_bce_loss(logits, bl)
    print("boundary bce:", bce.item())
