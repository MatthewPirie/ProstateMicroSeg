# ProstateMicroSeg/scripts/run_train_2d.py

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow: from src...

import argparse
import json
import subprocess
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data.dataset_2d import MicroUS2DSlicePatchDataset
from src.data.dataset_cases import MicroUSCaseDataset
from src.data.samplers_2d import OversampleForegroundBatchSampler
from src.models.monai_unet_2d import build_monai_unet_2d
from src.train.losses import CompoundBCEDiceLoss
from src.utils.metrics import dice_soft_from_logits, dice_hard_from_logits

from src.train.trainer_2d import train_one_epoch as train_one_epoch_2d
from src.train.trainer_2d import save_checkpoint as save_checkpoint_2d

from src.train.trainer_2d import validate_case_level_2d


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "NO_GIT_REPO"


def _make_run_dir(base_dir: str, run_name: str) -> Path:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    run_dir = (base / run_name) if run_name else (base / datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main() -> None:
    torch.set_num_threads(1)

    parser = argparse.ArgumentParser()

    # Data paths
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/pirie03/projects/aip-medilab/pirie03/ProstateMicroSeg/dataset/processed/Dataset120_MicroUSProstate",
        help="Processed dataset root with imagesTr/labelsTr/imagesTs/labelsTs (.npy).",
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        default="/home/pirie03/projects/aip-medilab/pirie03/ProstateMicroSeg/dataset/splits",
        help="Directory containing train.txt/val.txt/test.txt.",
    )

    # Case stats (optional)
    parser.add_argument("--use_case_stats", action="store_true", default=False)
    parser.add_argument(
        "--case_stats_path",
        type=str,
        default="",
        help="Path to JSON with per-case mean/std stats. If empty and --use_case_stats, uses <data_root>/case_stats.json.",
    )

    # Run / logging
    parser.add_argument("--runs_dir", type=str, default="runs")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--save_last", action="store_true", default=True)
    parser.add_argument("--no_save_last", action="store_false", dest="save_last")
    parser.add_argument("--save_best", action="store_true", default=True)
    parser.add_argument("--no_save_best", action="store_false", dest="save_best")
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--resume_ckpt", type=str, default="")

    # Training setup
    parser.add_argument("--model_variant", type=str, default="base", choices=["tiny", "small", "base"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--steps_per_epoch", type=int, default=588)
    parser.add_argument("--preprocess_mode", type=str, default="crop_pad")


    # Optimizer / LR schedule
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        choices=["none", "cosine", "polynomial"],
    )
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--momentum", type=float, default=0.99)
    parser.add_argument("--weight_decay", type=float, default=3e-5)

    # Dataloader / logging frequency
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=100)

    # Loss
    parser.add_argument("--w_bce", type=float, default=1.0)
    parser.add_argument("--w_dice", type=float, default=1.0)
    parser.add_argument("--batch_dice", action="store_true", default=True)
    parser.add_argument("--no_batch_dice", action="store_false", dest="batch_dice")

    # Preprocessing / shape
    parser.add_argument("--target_h", type=int, default=896)
    parser.add_argument("--target_w", type=int, default=1408)
    parser.add_argument("--transpose_hw", action="store_true")  # default False unless set

    args = parser.parse_args()

    # Resolve case stats path if requested
    data_root = Path(args.data_root)
    case_stats_path = args.case_stats_path.strip()
    if args.use_case_stats and case_stats_path == "":
        case_stats_path = str(data_root / "case_stats.json")

    # Run dir + TB
    run_dir = _make_run_dir(args.runs_dir, args.run_name)
    tb_dir = run_dir / "tb"
    writer = SummaryWriter(log_dir=str(tb_dir))

    print("Run dir:", run_dir, flush=True)

    cfg = vars(args).copy()
    with open(run_dir / "git_commit.txt", "w") as f:
        f.write(_get_git_commit() + "\n")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB", flush=True)

    target_hw = (int(args.target_h), int(args.target_w))
    steps_per_epoch = int(args.steps_per_epoch)
    total_steps = int(args.epochs) * steps_per_epoch
    preprocess_mode = args.preprocess_mode

    # Datasets
    train_ds = MicroUS2DSlicePatchDataset(
        dataset_root=args.data_root,
        splits_dir=args.splits_dir,
        split="train",
        target_hw=target_hw,
        transpose_hw=args.transpose_hw,
        seed=0,
        deterministic=False,
        do_augment=True,
        augment_seed=0,
        preprocess_mode=preprocess_mode
    )

    val_ds = MicroUSCaseDataset(
        dataset_root=args.data_root,
        splits_dir=args.splits_dir,
        split="val",
        use_case_stats=bool(args.use_case_stats),
        case_stats_path=case_stats_path if args.use_case_stats else None,
    )

    pin_memory = torch.cuda.is_available() and args.num_workers > 0

    train_batch_sampler = OversampleForegroundBatchSampler(
        dataset=train_ds,
        batch_size=args.batch_size,
        oversample_foreground_percent=0.33,
        seed=0,
        drop_last=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_batch_sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    # Case-level val loader MUST be batch_size=1
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    print(f"Train items (slices): {len(train_ds)} | Val items (cases): {len(val_ds)}", flush=True)
    print(f"Target HW: {target_hw} | Train batch size: {args.batch_size} | LR: {args.lr} | Epochs: {args.epochs}", flush=True)
    print(f"LR scheduler: {args.lr_scheduler}", flush=True)
    print(f"DataLoader: num_workers={args.num_workers}, pin_memory={pin_memory}", flush=True)

    # Model
    model, model_meta = build_monai_unet_2d(in_channels=1, out_channels=1, variant=args.model_variant)
    model = model.to(device)
    cfg["model"] = model_meta

    # Loss
    criterion = CompoundBCEDiceLoss(
        w_bce=args.w_bce,
        w_dice=args.w_dice,
        batch_dice=args.batch_dice,
    ).to(device)

    # Optimizer
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            nesterov=True,
            weight_decay=args.weight_decay,
        )

    # Scheduler (iteration-based)
    scheduler = None
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(total_steps, 1),
            eta_min=1e-6,
        )
    elif args.lr_scheduler == "polynomial":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: (1 - min(step, total_steps) / max(total_steps, 1)) ** 0.9,
        )

    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # Resume
    start_epoch = 1
    best_val = float("-inf")

    if args.resume_ckpt:
        ckpt_path = Path(args.resume_ckpt)
        ckpt = torch.load(ckpt_path, map_location=device)

        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1

        extra = ckpt.get("extra", {})
        if isinstance(extra, dict) and "best_val_dice_thr05" in extra:
            best_val = float(extra["best_val_dice_thr05"])

        print(f"Resumed from: {ckpt_path}", flush=True)
        print(f"Starting at epoch: {start_epoch}", flush=True)
        print(f"Current best_val_dice_thr05: {best_val}", flush=True)

    metrics_path = run_dir / "metrics.jsonl"

    # Training loop
    global_step = (start_epoch - 1) * steps_per_epoch

    for epoch in range(start_epoch, start_epoch + args.epochs):
        t_epoch0 = time.time()

        # Train (iteration-budgeted)
        train_metrics = train_one_epoch_2d(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            log_every=args.log_every,
            pin_memory=pin_memory,
            max_steps=steps_per_epoch,
            scheduler=scheduler,     # IMPORTANT: step per iteration inside train_one_epoch_2d
        )

        t1 = time.time()

        # Validate (case-level, deterministic crop/pad per slice to target_hw)
        val_metrics = validate_case_level_2d(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            dice_soft_fn=lambda lg, y: dice_soft_from_logits(lg, y, batch_dice=args.batch_dice),
            dice_hard_fn=dice_hard_from_logits,
            epoch=epoch,
            target_hw=target_hw,
            slice_batch_size=int(8),
            pin_memory=pin_memory,
        )

        t2 = time.time()

        # TensorBoard
        writer.add_scalar("train/total_loss", train_metrics["train_total_loss"], epoch)
        writer.add_scalar("val/total_loss", val_metrics["val_total_loss"], epoch)
        writer.add_scalar("val/dice_thr05", val_metrics["val_dice_thr05"], epoch)
        writer.flush()

        epoch_sec = time.time() - t_epoch0
        print(f"[Epoch {epoch}] time_sec={epoch_sec:.1f}", flush=True)
        print(f"[Epoch {epoch}] train_sec={t1-t_epoch0:.1f} val_sec={t2-t1:.1f} total_sec={t2-t_epoch0:.1f}", flush=True)

        record = {
            "epoch": epoch,
            "epoch_sec": epoch_sec,
            "train_total_loss": train_metrics["train_total_loss"],
            "val_total_loss": val_metrics["val_total_loss"],
            "val_dice_thr05": val_metrics["val_dice_thr05"],
        }
        with open(metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        # Checkpointing
        if args.save_last:
            save_checkpoint_2d(
                ckpt_path=run_dir / "checkpoint_last.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                extra={
                    "best_val_dice_thr05": best_val,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "args": vars(args),
                },
            )

        if args.save_best and val_metrics["val_dice_thr05"] > best_val:
            best_val = float(val_metrics["val_dice_thr05"])
            save_checkpoint_2d(
                ckpt_path=run_dir / "checkpoint_best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                extra={
                    "best_val_dice_thr05": best_val,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "args": vars(args),
                },
            )
            print(f"[Epoch {epoch}] New best VAL Dice(thr=0.5)={best_val:.4f} -> saved checkpoint_best.pt", flush=True)

        if args.save_every and (epoch % args.save_every == 0):
            save_checkpoint_2d(
                ckpt_path=run_dir / f"checkpoint_epoch{epoch:04d}.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                extra={"train_metrics": train_metrics, "val_metrics": val_metrics, "args": vars(args)},
            )

    print("Finished. Metrics:", metrics_path, flush=True)
    writer.close()


if __name__ == "__main__":
    main()
