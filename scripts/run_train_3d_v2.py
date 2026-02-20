# ProstateMicroSeg/scripts/run_train_3d_v2.py

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

from src.data.dataset_3d_v2 import MicroUS3DPatchDatasetV2
from src.data.dataset_cases import MicroUSCaseDataset
from src.data.samplers_3d_v2 import OversampleForegroundBatchSampler3DV2

from src.models.monai_unet_3d import build_monai_unet_3d
from src.train.losses import CompoundBCEDiceLoss
from src.utils.metrics import dice_soft_from_logits, dice_hard_from_logits

from src.train.trainer_3d_v2 import train_one_epoch_v2, validate_case_level_3d_v2, save_checkpoint_v2


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
    parser.add_argument(
        "--case_stats_path",
        type=str,
        default="",
        help="Path to case_stats.json. If empty, defaults to <data_root>/case_stats.json.",
    )
    parser.add_argument("--use_case_stats", action="store_true", default=True)
    parser.add_argument("--no_use_case_stats", action="store_false", dest="use_case_stats")

    # Run / logging
    parser.add_argument("--runs_dir", type=str, default="runs_3d_v2", help="Base directory for run outputs.")
    parser.add_argument("--run_name", type=str, default="", help="Optional run name (otherwise timestamp).")
    parser.add_argument("--save_last", action="store_true", default=True)
    parser.add_argument("--no_save_last", action="store_false", dest="save_last")
    parser.add_argument("--save_best", action="store_true", default=True)
    parser.add_argument("--no_save_best", action="store_false", dest="save_best")
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--resume_ckpt", type=str, default="")

    # Training setup
    parser.add_argument("--model_variant", type=str, default="nnunet_fullres", choices=["nnunet_fullres", "small"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--steps_per_epoch", type=int, default=250)
    parser.add_argument("--log_every", type=int, default=100)

    # Patch size (ZYX)
    parser.add_argument("--patch_z", type=int, default=14)
    parser.add_argument("--patch_y", type=int, default=256)
    parser.add_argument("--patch_x", type=int, default=448)

    # Foreground forcing
    parser.add_argument("--oversample_fg", type=float, default=0.33)
    parser.add_argument(
        "--oversample_mode",
        type=str,
        default="probabilistic",
        choices=["probabilistic", "deterministic"],
        help="How force_fg is assigned inside each batch.",
    )
    parser.add_argument("--fg_thr", type=float, default=0.5)

    # Optimizer / LR schedule
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.99)
    parser.add_argument("--weight_decay", type=float, default=3e-5)

    parser.add_argument("--lr_scheduler", type=str, default="polynomial", choices=["none", "cosine", "polynomial"])
    parser.add_argument("--poly_power", type=float, default=0.9)
    parser.add_argument("--min_lr", type=float, default=1e-6)

    # Loss
    parser.add_argument("--w_bce", type=float, default=1.0)
    parser.add_argument("--w_dice", type=float, default=1.0)
    parser.add_argument("--batch_dice", action="store_true", default=True)
    parser.add_argument("--no_batch_dice", action="store_false", dest="batch_dice")

    # Dataloader
    parser.add_argument("--num_workers", type=int, default=4)

    # Validation SW inference
    parser.add_argument("--val_overlap", type=float, default=0.5)
    parser.add_argument("--val_gaussian", action="store_true", default=True)
    parser.add_argument("--no_val_gaussian", action="store_false", dest="val_gaussian")
    parser.add_argument("--sw_batch_size", type=int, default=1)

    args = parser.parse_args()

    # ---- Run directory + metadata ----
    run_dir = _make_run_dir(args.runs_dir, args.run_name)
    tb_dir = run_dir / "tb"
    writer = SummaryWriter(log_dir=str(tb_dir))

    print("Run dir:", run_dir, flush=True)
    with open(run_dir / "git_commit.txt", "w") as f:
        f.write(_get_git_commit() + "\n")

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB", flush=True)

    pin_memory = torch.cuda.is_available() and args.num_workers > 0

    # ---- Paths / derived ----
    patch_zyx = (int(args.patch_z), int(args.patch_y), int(args.patch_x))
    case_stats_path = args.case_stats_path.strip() or str(Path(args.data_root) / "case_stats.json")

    total_steps = int(args.epochs) * int(args.steps_per_epoch)

    # ---- Datasets ----
    train_ds = MicroUS3DPatchDatasetV2(
        dataset_root=args.data_root,
        splits_dir=args.splits_dir,
        split="train",
        target_zyx=patch_zyx,
        seed=0,
        fg_threshold=args.fg_thr,
        deterministic=False,
        use_case_stats=args.use_case_stats,
        case_stats_path=case_stats_path,
    )

    val_ds = MicroUSCaseDataset(
        dataset_root=args.data_root,
        splits_dir=args.splits_dir,
        split="val",
        fg_threshold=args.fg_thr,
        use_case_stats=args.use_case_stats,
        case_stats_path=case_stats_path if args.use_case_stats else None,
    )

    # ---- Sampler + loaders ----
    train_batch_sampler = OversampleForegroundBatchSampler3DV2(
        dataset=train_ds,
        batch_size=args.batch_size,
        oversample_foreground_percent=args.oversample_fg,
        seed=0,
        drop_last=True,
        mode=args.oversample_mode,
        shuffle=True,
    )

    dl_kwargs = {}
    if args.num_workers > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_batch_sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        **dl_kwargs,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        **dl_kwargs,
    )

    print(f"Train cases: {len(train_ds)} | Val cases: {len(val_ds)}", flush=True)
    print(f"Patch ZYX: {patch_zyx} | Batch size: {args.batch_size} | Steps/epoch: {args.steps_per_epoch}", flush=True)
    print(f"Oversample FG: p={args.oversample_fg} | mode={args.oversample_mode}", flush=True)
    print(f"Optimizer: {args.optimizer} | LR: {args.lr} | Scheduler: {args.lr_scheduler}", flush=True)
    print(f"Val SW: overlap={args.val_overlap} gaussian={args.val_gaussian} sw_batch_size={args.sw_batch_size}", flush=True)

    # ---- Model ----
    model, model_meta = build_monai_unet_3d(in_channels=1, out_channels=1, variant=args.model_variant)
    model = model.to(device)

    # ---- Loss ----
    criterion = CompoundBCEDiceLoss(
        w_bce=args.w_bce,
        w_dice=args.w_dice,
        batch_dice=args.batch_dice,
    ).to(device)

    # ---- Optimizer ----
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            nesterov=True,
            weight_decay=args.weight_decay,
        )

    # ---- Scheduler (iteration-based, stepped inside trainer) ----
    scheduler = None
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(total_steps, 1),
            eta_min=float(args.min_lr),
        )
    elif args.lr_scheduler == "polynomial":
        power = float(args.poly_power)

        def _poly(step: int) -> float:
            s = min(step, max(total_steps, 1))
            return (1.0 - s / max(total_steps, 1)) ** power

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_poly)

    # ---- Save config ----
    cfg = vars(args).copy()
    cfg["model"] = model_meta
    cfg["patch_zyx"] = patch_zyx
    cfg["case_stats_path"] = case_stats_path
    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # ---- Resume (optional) ----
    start_epoch = 1
    best_val = float("-inf")
    global_step = 0  # for scheduler bookkeeping if needed later

    if args.resume_ckpt:
        ckpt_path = Path(args.resume_ckpt)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        extra = ckpt.get("extra", {})
        if isinstance(extra, dict) and "best_val_dice_thr05" in extra:
            best_val = float(extra["best_val_dice_thr05"])
        if isinstance(extra, dict) and "global_step" in extra:
            global_step = int(extra["global_step"])
        print(f"Resumed from: {ckpt_path}", flush=True)
        print(f"Starting at epoch: {start_epoch}", flush=True)
        print(f"Current best_val_dice_thr05: {best_val}", flush=True)

    # ---- Training loop ----
    metrics_path = run_dir / "metrics.jsonl"

    for epoch in range(start_epoch, start_epoch + int(args.epochs)):
        t0 = time.time()

        train_metrics = train_one_epoch_v2(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            steps_per_epoch=int(args.steps_per_epoch),
            log_every=int(args.log_every),
            pin_memory=pin_memory,
            scheduler=scheduler,  # stepped per-iteration in trainer
        )

        t1 = time.time()

        val_metrics = validate_case_level_3d_v2(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            dice_soft_fn=lambda lg, y: dice_soft_from_logits(lg, y, batch_dice=args.batch_dice),
            dice_hard_fn=dice_hard_from_logits,
            epoch=epoch,
            roi_size=patch_zyx,
            overlap=float(args.val_overlap),
            sw_batch_size=int(args.sw_batch_size),
            gaussian=bool(args.val_gaussian),
            pin_memory=pin_memory,
        )

        t2 = time.time()

        # TensorBoard
        writer.add_scalar("train/total_loss", train_metrics["train_total_loss"], epoch)
        writer.add_scalar("train/bce", train_metrics["train_bce"], epoch)
        writer.add_scalar("train/dice_loss", train_metrics["train_dice_loss"], epoch)

        writer.add_scalar("val/total_loss", val_metrics["val_total_loss"], epoch)
        writer.add_scalar("val/dice_thr05", val_metrics["val_dice_thr05"], epoch)
        writer.add_scalar("val/dice_soft", val_metrics["val_dice_soft"], epoch)
        writer.flush()

        epoch_sec = time.time() - t0
        print(f"[Epoch {epoch}] time_sec={epoch_sec:.1f} train_sec={t1-t0:.1f} val_sec={t2-t1:.1f}", flush=True)

        record = {
            "epoch": epoch,
            "epoch_sec": epoch_sec,
            "train_total_loss": train_metrics["train_total_loss"],
            "train_bce": train_metrics["train_bce"],
            "train_dice_loss": train_metrics["train_dice_loss"],
            "val_total_loss": val_metrics["val_total_loss"],
            "val_dice_thr05": val_metrics["val_dice_thr05"],
            "val_dice_soft": val_metrics["val_dice_soft"],
        }
        with open(metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        # Checkpoints
        if args.save_last:
            save_checkpoint_v2(
                ckpt_path=run_dir / "checkpoint_last.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                extra={
                    "best_val_dice_thr05": best_val,
                    "global_step": global_step,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "args": vars(args),
                },
            )

        if args.save_best and val_metrics["val_dice_thr05"] > best_val:
            best_val = val_metrics["val_dice_thr05"]
            save_checkpoint_v2(
                ckpt_path=run_dir / "checkpoint_best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                extra={
                    "best_val_dice_thr05": best_val,
                    "global_step": global_step,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "args": vars(args),
                },
            )
            print(f"[Epoch {epoch}] New best VAL Dice(thr=0.5)={best_val:.4f} -> saved checkpoint_best.pt", flush=True)

        if args.save_every and (epoch % int(args.save_every) == 0):
            save_checkpoint_v2(
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
