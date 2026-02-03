# ProstateMicroSeg/src/train/trainer_2d.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch


def train_one_epoch(
    model: torch.nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    criterion,
    device: torch.device,
    epoch: int,
    log_every: int = 10,
    pin_memory: bool = False,
) -> Dict[str, float]:
    """
    One training epoch.

    Returns:
        {"train_loss": float}
    """
    model.train()

    running_loss = 0.0
    loss_sum = 0.0
    n_steps = 0

    for step, batch in enumerate(train_loader, start=1):
        imgs = batch["image"].to(device, non_blocking=pin_memory)
        lbls = batch["label"].to(device, non_blocking=pin_memory)

        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = criterion(logits, lbls)

        loss.backward()
        optimizer.step()

        loss_val = float(loss.item())
        running_loss += loss_val
        loss_sum += loss_val
        n_steps += 1

        if step == 1 or (log_every > 0 and step % log_every == 0):
            avg_loss = running_loss / (1 if step == 1 else log_every)
            print(f"[Epoch {epoch}] step {step}/{len(train_loader)} | train_loss={avg_loss:.4f}")
            running_loss = 0.0

    return {"train_loss": loss_sum / max(n_steps, 1)}


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    dice_fn,
    epoch: int,
    pin_memory: bool = False,
) -> Dict[str, float]:
    """
    Validation loop.

    dice_fn signature:
        dice_fn(logits: Tensor, targets: Tensor) -> Tensor (scalar)

    Returns:
        {"val_dice": float}
    """
    model.eval()

    dice_sum = 0.0
    n_batches = 0

    for batch in val_loader:
        imgs = batch["image"].to(device, non_blocking=pin_memory)
        lbls = batch["label"].to(device, non_blocking=pin_memory)

        logits = model(imgs)
        dice = dice_fn(logits, lbls)

        dice_sum += float(dice.item())
        n_batches += 1

    val_dice = dice_sum / max(n_batches, 1)
    print(f"[Epoch {epoch}] VAL Dice={val_dice:.4f}")
    return {"val_dice": val_dice}


def save_checkpoint(
    ckpt_path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    extra: Optional[dict] = None,
) -> None:
    """
    Minimal checkpoint saver.
    """
    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if extra:
        payload.update(extra)

    torch.save(payload, ckpt_path)
