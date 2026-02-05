# src/train/losses.py

import torch
import torch.nn as nn


def soft_dice_loss(probs, targets, batch_dice=True, eps=1e-6):
    """
    probs, targets: (B,1,H,W) in [0,1]
    """

    probs = probs.contiguous()
    targets = targets.contiguous()

    if batch_dice:
        # treat batch as one big volume
        inter = (probs * targets).sum()
        denom = probs.sum() + targets.sum()
        dice = (2 * inter + eps) / (denom + eps)
        return 1 - dice

    else:
        # per-sample Dice
        B = probs.shape[0]
        probs = probs.view(B, -1)
        targets = targets.view(B, -1)

        inter = (probs * targets).sum(dim=1)
        denom = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2 * inter + eps) / (denom + eps)

        return 1 - dice.mean()


class CompoundBCEDiceLoss(nn.Module):
    def __init__(self, w_bce=1.0, w_dice=1.0, batch_dice=True):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.w_bce = w_bce
        self.w_dice = w_dice
        self.batch_dice = batch_dice

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        dice = soft_dice_loss(probs, targets, batch_dice=self.batch_dice)

        total = self.w_bce * bce + self.w_dice * dice

        return total, {
            "bce": bce.detach(),
            "dice_loss": dice.detach(),
        }
