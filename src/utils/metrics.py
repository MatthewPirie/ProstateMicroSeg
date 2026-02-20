# ProstateMicroSeg/src/utils/metrics.py

from __future__ import annotations
import torch

def dice_hard_from_logits(logits, targets, threshold=0.5, eps=1e-6):
    if logits.shape != targets.shape:
        raise RuntimeError(f"Shape mismatch: logits={tuple(logits.shape)} targets={tuple(targets.shape)}")

    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = (targets > 0.5).float()

    reduce_dims = tuple(range(2, preds.ndim))  # works for 2D or 3D
    inter = (preds * targets).sum(dim=reduce_dims)
    denom = preds.sum(dim=reduce_dims) + targets.sum(dim=reduce_dims)

    dice = (2 * inter + eps) / (denom + eps)
    return dice.mean()

def dice_soft_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    batch_dice: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Soft Dice metric (no threshold).

    This matches the Dice used inside Dice loss, but as a metric.

    Args:
        logits:  (B,1,H,W)
        targets: (B,1,H,W)
        batch_dice: if True compute Dice over full batch

    Returns:
        scalar Dice
    """
    probs = torch.sigmoid(logits)

    if batch_dice:
        inter = (probs * targets).sum()
        denom = probs.sum() + targets.sum()
        dice = (2 * inter + eps) / (denom + eps)
        return dice

    else:
        B = probs.shape[0]
        probs = probs.view(B, -1)
        targets = targets.view(B, -1)

        inter = (probs * targets).sum(dim=1)
        denom = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2 * inter + eps) / (denom + eps)
        return dice.mean()
