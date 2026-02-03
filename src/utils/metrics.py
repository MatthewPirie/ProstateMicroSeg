# ProstateMicroSeg/src/utils/metrics.py

from __future__ import annotations

import torch


def dice_score_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute mean Dice over a batch from raw logits.

    Args:
        logits:  (B, 1, H, W) raw model outputs (logits)
        targets: (B, 1, H, W) binary {0,1} targets
        threshold: probability threshold after sigmoid
        eps: numerical stability

    Returns:
        scalar torch.Tensor (mean Dice over batch)
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(preds.shape[0], -1)
    targets = targets.view(targets.shape[0], -1)

    inter = (preds * targets).sum(dim=1)
    denom = preds.sum(dim=1) + targets.sum(dim=1)

    dice = (2 * inter + eps) / (denom + eps)
    return dice.mean()
