# ProstateMicroSeg/scripts/run_train_2d.py

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow: from src...

import argparse
import json
import subprocess
from datetime import datetime
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data.dataset_2d import MicroUS2DSliceDataset
from src.models.monai_unet_2d import build_monai_unet_2d
from src.train.trainer_2d import train_one_epoch, validate, save_checkpoint
from src.utils.metrics import dice_score_from_logits


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "NO_GIT_REPO"


def _make_run_dir(base_dir: str, run_name: str) -> Path:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    if run_name:
        run_dir = base / run_name
    else:
        run_dir = base / datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()

    # --- data ---
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/pirie03/projects/aip-medilab/pirie03/ProstateMicroSeg/dataset/processed/Dataset120_MicroUSProstate",
        help="Processed dataset root containing imagesTr/labelsTr/imagesTs/labelsTs with .npy files",
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        default="/home/pirie03/projects/aip-medilab/pirie03/ProstateMicroSeg/dataset/splits",
        help="Directory containing train.txt/val.txt/test.txt",
    )

    # --- run / logging ---
    parser.add_argument("--runs_dir", type=str, default="runs", help="Base directory for run outputs")
    parser.add_argument("--run_name", type=str, default="", help="Optional run name (otherwise timestamp)")
    parser.add_argument("--save_last", action="store_true", default=True, help="Save last checkpoint each epoch")
    parser.add_argument("--no_save_last", action="store_false", dest="save_last", help="Disable saving last checkpoint")
    parser.add_argument("--save_best", action="store_true", default=True, help="Save best checkpoint by val dice")
    parser.add_argument("--no_save_best", action="store_false", dest="save_best", help="Disable saving best checkpoint")
    parser.add_argument("--save_every", type=int, default=0, help="If >0, also save checkpoint every N epochs")
    parser.add_argument("--resume_ckpt", type=str, default="", help="Path to checkpoint (.pt) to resume from (loads model+optimizer+epoch)")
    # --- training ---
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=10)
    

    # --- preprocessing / shape ---
    parser.add_argument("--target_h", type=int, default=896)
    parser.add_argument("--target_w", type=int, default=1408)
    parser.add_argument("--transpose_hw", action="store_true")  # default False unless set
    parser.add_argument("--only_foreground_slices", action="store_true")

    args = parser.parse_args()

    # ---- run dir + metadata ----
    run_dir = _make_run_dir(args.runs_dir, args.run_name)
    tb_dir = run_dir / "tb"
    writer = SummaryWriter(log_dir=str(tb_dir))


    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    with open(run_dir / "git_commit.txt", "w") as f:
        f.write(_get_git_commit() + "\n")

    print("Run dir:", run_dir)

    # ---- device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    target_hw = (args.target_h, args.target_w)

    # ---- data ----
    train_ds = MicroUS2DSliceDataset(
        dataset_root=args.data_root,
        splits_dir=args.splits_dir,
        split="train",
        target_hw=target_hw,
        transpose_hw=args.transpose_hw,
        only_foreground_slices=args.only_foreground_slices,
    )
    val_ds = MicroUS2DSliceDataset(
        dataset_root=args.data_root,
        splits_dir=args.splits_dir,
        split="val",
        target_hw=target_hw,
        transpose_hw=args.transpose_hw,
        only_foreground_slices=args.only_foreground_slices,
    )

    pin_memory = torch.cuda.is_available() and args.num_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    print(f"Train slices: {len(train_ds)} | Val slices: {len(val_ds)}")
    print(f"Target HW: {target_hw} | Batch size: {args.batch_size} | LR: {args.lr} | Epochs: {args.epochs}")
    print(f"DataLoader: num_workers={args.num_workers}, pin_memory={pin_memory}")

    # ---- model ----
    model = build_monai_unet_2d(in_channels=1, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ---- training loop (orchestration) ----
    start_epoch = 1
    best_val = float("-inf")

    # ---- resume (optional) ----
    if args.resume_ckpt:
        ckpt_path = Path(args.resume_ckpt)
        ckpt = torch.load(ckpt_path, map_location=device)

        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # resume epoch count
        start_epoch = int(ckpt.get("epoch", 0)) + 1

        # if you saved best_val_dice in "extra", restore it
        extra = ckpt.get("extra", {})
        if isinstance(extra, dict) and "best_val_dice" in extra:
            best_val = float(extra["best_val_dice"])

        print(f"Resumed from: {ckpt_path}", flush=True)
        print(f"Starting at epoch: {start_epoch}", flush=True)
        print(f"Current best_val: {best_val}", flush=True)

    metrics_path = run_dir / "metrics.jsonl"

    for epoch in range(start_epoch, start_epoch + args.epochs):
        t_epoch0 = time.time()

        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            log_every=args.log_every,
            pin_memory=pin_memory,
        )
        val_metrics = validate(
            model=model,
            val_loader=val_loader,
            device=device,
            dice_fn=dice_score_from_logits,
            epoch=epoch,
            pin_memory=pin_memory,
        )

        writer.add_scalar("train/loss", train_metrics["train_loss"], epoch)
        writer.add_scalar("val/dice", val_metrics["val_dice"], epoch)
        writer.flush()

        epoch_sec = time.time() - t_epoch0
        print(f"[Epoch {epoch}] time_sec={epoch_sec:.1f}", flush=True)

        # write one line per epoch
        record = {"epoch": epoch, "epoch_sec": epoch_sec, **train_metrics, **val_metrics}
        with open(metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        # save last
        if args.save_last:
            save_checkpoint(
                ckpt_path=run_dir / "checkpoint_last.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                extra={"best_val_dice": best_val, "train_metrics": train_metrics, "val_metrics": val_metrics, "args": vars(args)},
            )

        # save best
        if args.save_best and val_metrics["val_dice"] > best_val:
            best_val = val_metrics["val_dice"]
            save_checkpoint(
                ckpt_path=run_dir / "checkpoint_best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                extra={"best_val_dice": best_val, "train_metrics": train_metrics, "val_metrics": val_metrics, "args": vars(args)},
            )
            print(f"[Epoch {epoch}] New best VAL Dice={best_val:.4f} -> saved checkpoint_best.pt")

        # optional periodic save
        if args.save_every and (epoch % args.save_every == 0):
            save_checkpoint(
                ckpt_path=run_dir / f"checkpoint_epoch{epoch:04d}.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                extra={"train_metrics": train_metrics, "val_metrics": val_metrics, "args": vars(args)},
            )

    print("Finished. Metrics:", metrics_path)
    writer.close()


if __name__ == "__main__":
    main()
