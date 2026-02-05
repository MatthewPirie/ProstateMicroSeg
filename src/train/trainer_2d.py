# ProstateMicroSeg/src/train/trainer_2d.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch


def train_one_epoch(
    model: torch.nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    criterion,  # expects: total_loss, parts = criterion(logits, targets)
    device: torch.device,
    epoch: int,
    log_every: int = 100,
    pin_memory: bool = False,
) -> Dict[str, float]:
    """
    One training epoch.

    Returns:
        {
          "train_total_loss": float,
          "train_bce": float,
          "train_dice_loss": float
        }
    """
    model.train()

    # For console logging (windowed average)
    running_total = 0.0
    running_steps = 0

    # Epoch averages
    total_sum = 0.0
    bce_sum = 0.0
    dice_loss_sum = 0.0
    n_steps = 0

    for step, batch in enumerate(train_loader, start=1):
        imgs = batch["image"].to(device, non_blocking=pin_memory)
        lbls = batch["label"].to(device, non_blocking=pin_memory)

        optimizer.zero_grad(set_to_none=True)

        logits = model(imgs)
        total_loss, parts = criterion(logits, lbls)

        # parts are raw tensors (not detached)
        bce = parts["bce"]
        dice_loss = parts["dice_loss"]

        total_loss.backward()
        optimizer.step()

        # Convert to python floats for aggregation
        total_val = float(total_loss.item())
        bce_val = float(bce.detach().item())
        dice_loss_val = float(dice_loss.detach().item())

        total_sum += total_val
        bce_sum += bce_val
        dice_loss_sum += dice_loss_val
        n_steps += 1

        running_total += total_val
        running_steps += 1

        if step == 1 or (log_every > 0 and step % log_every == 0):
            avg_total = running_total / max(running_steps, 1)
            print(
                f"[Epoch {epoch}] step {step}/{len(train_loader)} | "
                f"train_total_loss={avg_total:.4f}"
            )
            running_total = 0.0
            running_steps = 0

    return {
        "train_total_loss": total_sum / max(n_steps, 1),
        "train_bce": bce_sum / max(n_steps, 1),
        "train_dice_loss": dice_loss_sum / max(n_steps, 1),
    }


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    val_loader,
    criterion,  # expects: total_loss, parts = criterion(logits, targets)
    device: torch.device,
    dice_soft_fn,  # expects: dice_soft_fn(logits, targets) -> scalar tensor
    dice_hard_fn,  # expects: dice_hard_fn(logits, targets) -> scalar tensor
    epoch: int,
    pin_memory: bool = False,
) -> Dict[str, float]:
    """
    Validation loop.

    Returns:
        {
          "val_total_loss": float,
          "val_bce": float,
          "val_dice_loss": float,
          "val_dice_soft": float,
          "val_dice_thr05": float
        }
    """
    model.eval()

    total_sum = 0.0
    bce_sum = 0.0
    dice_loss_sum = 0.0
    dice_soft_sum = 0.0
    dice_hard_sum = 0.0
    n_batches = 0

    for batch in val_loader:
        imgs = batch["image"].to(device, non_blocking=pin_memory)
        lbls = batch["label"].to(device, non_blocking=pin_memory)

        logits = model(imgs)

        total_loss, parts = criterion(logits, lbls)
        bce = parts["bce"]
        dice_loss = parts["dice_loss"]

        dice_soft = dice_soft_fn(logits, lbls)
        dice_hard = dice_hard_fn(logits, lbls)

        total_sum += float(total_loss.item())
        bce_sum += float(bce.detach().item())
        dice_loss_sum += float(dice_loss.detach().item())
        dice_soft_sum += float(dice_soft.detach().item())
        dice_hard_sum += float(dice_hard.detach().item())

        n_batches += 1

    val_total = total_sum / max(n_batches, 1)
    val_bce = bce_sum / max(n_batches, 1)
    val_dice_loss = dice_loss_sum / max(n_batches, 1)
    val_dice_soft = dice_soft_sum / max(n_batches, 1)
    val_dice_thr05 = dice_hard_sum / max(n_batches, 1)

    print(
        f"[Epoch {epoch}] "
        f"VAL total_loss={val_total:.4f} | "
        f"VAL dice_thr05={val_dice_thr05:.4f}"
    )

    return {
        "val_total_loss": val_total,
        "val_bce": val_bce,
        "val_dice_loss": val_dice_loss,
        "val_dice_soft": val_dice_soft,
        "val_dice_thr05": val_dice_thr05,
    }


def save_checkpoint(
    ckpt_path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    extra: Optional[dict] = None,
) -> None:
    """
    Minimal checkpoint saver.
    Stores auxiliary info under payload["extra"] to avoid key collisions.
    """
    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "extra": extra if extra is not None else {},
    }

    torch.save(payload, ckpt_path)
