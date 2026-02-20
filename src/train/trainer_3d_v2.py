# src/train/trainer_3d_v2.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch


def train_one_epoch_v2(
    model: torch.nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    criterion,  # expects: total_loss, parts = criterion(logits, targets)
    device: torch.device,
    epoch: int,
    steps_per_epoch: int = 250,      # nnU-Net-style fixed optimizer updates per epoch
    log_every: int = 50,
    pin_memory: bool = False,
    scheduler=None,                 # if provided, step per-iteration
) -> Dict[str, float]:
    """
    One training epoch with a fixed number of optimizer steps.

    Key behavior:
      - We iterate steps_per_epoch times, regardless of dataset length.
      - If the loader iterator runs out (StopIteration), we re-create it and keep going.
      - This matches the nnU-Net "epoch = fixed number of iterations" idea.
    """
    model.train()

    total_sum = 0.0
    bce_sum = 0.0
    dice_loss_sum = 0.0

    running_total = 0.0
    running_steps = 0

    it = iter(train_loader)

    for step in range(1, steps_per_epoch + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)

        imgs = batch["image"].to(device, non_blocking=pin_memory)
        lbls = batch["label"].to(device, non_blocking=pin_memory)

        optimizer.zero_grad(set_to_none=True)

        logits = model(imgs)
        total_loss, parts = criterion(logits, lbls)

        total_loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_val = float(total_loss.item())
        bce_val = float(parts["bce"].detach().item())
        dice_loss_val = float(parts["dice_loss"].detach().item())

        total_sum += total_val
        bce_sum += bce_val
        dice_loss_sum += dice_loss_val

        running_total += total_val
        running_steps += 1

        if step == 1 or (log_every > 0 and step % log_every == 0):
            avg_total = running_total / max(running_steps, 1)
            print(f"[Epoch {epoch}] step {step}/{steps_per_epoch} | train_total_loss={avg_total:.4f}", flush=True)
            running_total = 0.0
            running_steps = 0

    return {
        "train_total_loss": total_sum / max(steps_per_epoch, 1),
        "train_bce": bce_sum / max(steps_per_epoch, 1),
        "train_dice_loss": dice_loss_sum / max(steps_per_epoch, 1),
    }


@torch.no_grad()
def validate_case_level_3d_v2(
    model: torch.nn.Module,
    val_loader,              # expects full volumes, batch_size=1
    criterion,
    device: torch.device,
    dice_soft_fn,
    dice_hard_fn,
    epoch: int,
    roi_size: Tuple[int, int, int] = (14, 256, 448),
    overlap: float = 0.5,
    sw_batch_size: int = 1,
    gaussian: bool = True,
    pin_memory: bool = False,
) -> Dict[str, float]:
    """
    Case-level 3D validation using sliding-window inference (MONAI).

    Expects:
      image: [1,1,Z,Y,X]
      label: [1,1,Z,Y,X]
    """
    from monai.inferers import sliding_window_inference

    model.eval()

    total_sum = 0.0
    bce_sum = 0.0
    dice_loss_sum = 0.0
    dice_soft_sum = 0.0
    dice_hard_sum = 0.0
    n_cases = 0

    mode = "gaussian" if gaussian else "constant"

    for batch in val_loader:
        img = batch["image"].to(device, non_blocking=pin_memory)
        lbl = batch["label"].to(device, non_blocking=pin_memory)

        if img.ndim != 5:
            raise RuntimeError(f"Expected img [B,1,Z,Y,X], got {tuple(img.shape)}")
        if img.shape[0] != 1 or img.shape[1] != 1:
            raise RuntimeError(f"Expected batch_size=1 and C=1, got {tuple(img.shape)}")

        logits = sliding_window_inference(
            inputs=img,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=float(overlap),
            mode=mode,
        )

        total_loss, parts = criterion(logits, lbl)
        dice_soft = dice_soft_fn(logits, lbl)
        dice_hard = dice_hard_fn(logits, lbl)

        total_sum += float(total_loss.item())
        bce_sum += float(parts["bce"].detach().item())
        dice_loss_sum += float(parts["dice_loss"].detach().item())
        dice_soft_sum += float(dice_soft.detach().item())
        dice_hard_sum += float(dice_hard.detach().item())
        n_cases += 1

    out = {
        "val_total_loss": total_sum / max(n_cases, 1),
        "val_bce": bce_sum / max(n_cases, 1),
        "val_dice_loss": dice_loss_sum / max(n_cases, 1),
        "val_dice_soft": dice_soft_sum / max(n_cases, 1),
        "val_dice_thr05": dice_hard_sum / max(n_cases, 1),
    }

    print(
        f"[Epoch {epoch}] VAL(case,3D) total_loss={out['val_total_loss']:.4f} | "
        f"dice_thr05={out['val_dice_thr05']:.4f}",
        flush=True,
    )
    return out


def save_checkpoint_v2(
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
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "extra": extra if extra is not None else {},
    }
    torch.save(payload, ckpt_path)
