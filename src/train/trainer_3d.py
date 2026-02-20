
# ProstateMicroSeg/src/train/trainer_3d.py

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
    max_steps: int = 250,   # nnU-Net-style: fixed optimizer steps per epoch
    scheduler=None,         # if provided, step per iteration (optional)
) -> Dict[str, float]:
    """
    One training epoch with a fixed number of optimizer steps (nnU-Net-style).

    Works for both 2D and 3D because it only assumes tensors shaped:
      - images: [B, C, ...]
      - labels: [B, C, ...]
    """
    model.train()

    running_total = 0.0
    running_steps = 0

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

        total_loss.backward()
        optimizer.step()

        # If you want iteration-based scheduling, uncomment this and remove scheduler.step() in run_train
        # if scheduler is not None:
        #     scheduler.step()

        total_val = float(total_loss.item())
        bce_val = float(parts["bce"].detach().item())
        dice_loss_val = float(parts["dice_loss"].detach().item())

        total_sum += total_val
        bce_sum += bce_val
        dice_loss_sum += dice_loss_val
        n_steps += 1

        running_total += total_val
        running_steps += 1

        if step == 1 or (log_every > 0 and step % log_every == 0):
            avg_total = running_total / max(running_steps, 1)
            print(f"[Epoch {epoch}] step {step}/{max_steps} | train_total_loss={avg_total:.4f}")
            running_total = 0.0
            running_steps = 0

        if max_steps is not None and step >= max_steps:
            break

    return {
        "train_total_loss": total_sum / max(n_steps, 1),
        "train_bce": bce_sum / max(n_steps, 1),
        "train_dice_loss": dice_loss_sum / max(n_steps, 1),
    }


@torch.no_grad()
def validate_case_level_3d(
    model: torch.nn.Module,
    val_loader,              # MicroUSCaseDataset, batch_size=1
    criterion,
    device: torch.device,
    dice_soft_fn,
    dice_hard_fn,
    epoch: int,
    roi_size: Tuple[int, int, int] = (14, 256, 448),  # your training patch (Z,Y,X)
    overlap: float = 0.0,
    sw_batch_size: int = 1,
    gaussian: bool = False,
    pin_memory: bool = False,
) -> Dict[str, float]:
    """
    Case-level validation for a 3D model using sliding-window inference.

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
        img = batch["image"].to(device, non_blocking=pin_memory)  # [1,1,Z,Y,X]
        lbl = batch["label"].to(device, non_blocking=pin_memory)  # [1,1,Z,Y,X]

        if img.ndim != 5:
            raise RuntimeError(f"Expected img [B,1,Z,Y,X], got {tuple(img.shape)}")
        if img.shape[0] != 1 or img.shape[1] != 1:
            raise RuntimeError(f"Expected batch_size=1 and C=1, got {tuple(img.shape)}")

        # roi_size = (14, 256, 448)
        # overlap = 0.5
        # sw_batch_size = 1

        # # 1) run sliding window
        # logits_sw = sliding_window_inference(
        #     inputs=img,
        #     roi_size=roi_size,
        #     sw_batch_size=sw_batch_size,
        #     predictor=model,
        #     overlap=overlap,
        #     mode="gaussian",
        # )

        # # 2) pick a fixed patch location (CENTER PATCH)
        # Zc, Yc, Xc = roi_size
        # zc = img.shape[2] // 2
        # yc = img.shape[3] // 2
        # xc = img.shape[4] // 2

        # z0 = zc - Zc // 2
        # y0 = yc - Yc // 2
        # x0 = xc - Xc // 2

        # img_patch = img[:, :, z0:z0+Zc, y0:y0+Yc, x0:x0+Xc]

        # # 3) run model directly on that patch
        # logits_direct = model(img_patch)

        # # 4) crop the same region out of the SW output
        # logits_sw_crop = logits_sw[:, :, z0:z0+Zc, y0:y0+Yc, x0:x0+Xc]

        # # 5) compare
        # max_abs_diff = (logits_direct - logits_sw_crop).abs().max().item()
        # mean_abs_diff = (logits_direct - logits_sw_crop).abs().mean().item()

        # print("SW vs DIRECT logits diff:",
        #     "max", max_abs_diff,
        #     "mean", mean_abs_diff,
        #     flush=True)

        # # Optional: dice on that patch only (same region, same label)
        # lbl_patch = lbl[:, :, z0:z0+Zc, y0:y0+Yc, x0:x0+Xc]
        # dice_patch = dice_hard_fn(logits_direct, lbl_patch).item()
        # dice_patch_sw = dice_hard_fn(logits_sw_crop, lbl_patch).item()
        # print("Patch dice direct:", dice_patch, "Patch dice sw:", dice_patch_sw, flush=True)

        # break

        # Sliding window over full volume, stitched logits returned with full-size shape
        logits = sliding_window_inference(
            inputs=img,
            roi_size=roi_size,           # (14,256,448)
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=overlap,
            mode=mode,
        )

        gt_frac = (lbl > 0.5).float().mean().item()
        pred_frac = (torch.sigmoid(logits) > 0.5).float().mean().item()
        print("GT fg fraction:", gt_frac, "Pred fg fraction:", pred_frac, flush=True)

        total_loss, parts = criterion(logits, lbl)
        dice_soft = dice_soft_fn(logits, lbl)
        dice_hard = dice_hard_fn(logits, lbl)

        total_sum += float(total_loss.item())
        bce_sum += float(parts["bce"].detach().item())
        dice_loss_sum += float(parts["dice_loss"].detach().item())
        dice_soft_sum += float(dice_soft.detach().item())
        dice_hard_sum += float(dice_hard.detach().item())
        n_cases += 1

        print("img", tuple(img.shape), "lbl", tuple(lbl.shape), "logits", tuple(logits.shape), flush=True)

    out = {
        "val_total_loss": total_sum / max(n_cases, 1),
        "val_bce": bce_sum / max(n_cases, 1),
        "val_dice_loss": dice_loss_sum / max(n_cases, 1),
        "val_dice_soft": dice_soft_sum / max(n_cases, 1),
        "val_dice_thr05": dice_hard_sum / max(n_cases, 1),
    }

    print(
        f"[Epoch {epoch}] VAL(case,3D) total_loss={out['val_total_loss']:.4f} | "
        f"dice_thr05={out['val_dice_thr05']:.4f}"
    )
    return out

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


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    val_loader,
    criterion,
    device: torch.device,
    dice_soft_fn,
    dice_hard_fn,
    epoch: int,
    pin_memory: bool = False,
) -> Dict[str, float]:
    """
    Validation loop (works for 2D or 3D).
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

