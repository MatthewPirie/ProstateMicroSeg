# scripts/tests/viz_pred_3d.py

import sys
sys.path.append("/home/pirie03/projects/aip-medilab/pirie03/ProstateMicroSeg")

from pathlib import Path
import json

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.models.monai_unet_3d import build_monai_unet_3d
from src.data.dataset_cases import MicroUSCaseDataset

RUN_DIR = Path("/home/pirie03/projects/aip-medilab/pirie03/ProstateMicroSeg/runs_3d/20260212_121431")
CKPT_PATH = RUN_DIR / "checkpoint_best.pt"

DATA_ROOT = Path("/home/pirie03/projects/aip-medilab/pirie03/ProstateMicroSeg/dataset/processed/Dataset120_MicroUSProstate")
SPLITS_DIR = Path("/home/pirie03/projects/aip-medilab/pirie03/ProstateMicroSeg/dataset/splits")
CASE_STATS_PATH = DATA_ROOT / "case_stats.json"


def _to_01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    lo = float(np.min(x))
    hi = float(np.max(x))
    return (x - lo) / (hi - lo + 1e-8)


def _bright_overlay(img01: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Grayscale overlay: brighten masked pixels.
    img01: [Y,X] in [0,1]
    mask:  [Y,X] in {0,1}
    """
    out = img01.copy()
    m = mask > 0.5
    out[m] = np.clip(out[m] * 0.5 + 0.5, 0, 1)
    return out


@torch.no_grad()
def main() -> None:
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Missing checkpoint: {CKPT_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)

    ckpt = torch.load(CKPT_PATH, map_location=device)

    # Read run config if present (to auto-pick model_variant + roi_size)
    cfg_path = RUN_DIR / "config.json"
    cfg = {}
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            cfg = json.load(f)

    # model variant
    variant = cfg.get("model_variant", "nnunet_fullres")
    print("Model variant:", variant, flush=True)

    # roi_size: prefer your training patch sizes if present in config
    # (roi_size must be (Z,Y,X) for MONAI sliding_window_inference in 3D)
    patch_z = int(cfg.get("patch_z", 14))
    patch_y = int(cfg.get("patch_y", 256))
    patch_x = int(cfg.get("patch_x", 448))
    roi_size = (patch_z, patch_y, patch_x)
    print("ROI size (Z,Y,X):", roi_size, flush=True)

    # build + load model
    model, _meta = build_monai_unet_3d(in_channels=1, out_channels=1, variant=variant)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    # load one validation case (full volume)
    val_ds = MicroUSCaseDataset(
        dataset_root=str(DATA_ROOT),
        splits_dir=str(SPLITS_DIR),
        split="val",
        use_case_stats=True,
        case_stats_path=str(CASE_STATS_PATH),
    )
    sample = val_ds[0]
    case_id = sample["case_id"]
    img = sample["image"].unsqueeze(0).to(device)  # [1,1,Z,Y,X]
    lbl = sample["label"].unsqueeze(0).to(device)  # [1,1,Z,Y,X]

    # sliding window inference (full-volume logits)
    from monai.inferers import sliding_window_inference

    # Safe defaults for viz (not trying to be ultra-fast here)
    overlap = 0.5
    sw_batch_size = 1
    mode = "gaussian"

    logits = sliding_window_inference(
        inputs=img,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=overlap,
        mode=mode,
    )

    prob = torch.sigmoid(logits)
    pred = (prob > 0.5).float()

    # move to cpu numpy for plotting
    img_np = img[0, 0].detach().cpu().numpy()   # [Z,Y,X]
    lbl_np = lbl[0, 0].detach().cpu().numpy()   # [Z,Y,X]
    pred_np = pred[0, 0].detach().cpu().numpy() # [Z,Y,X]

    Z = img_np.shape[0]
    slice_ids = [
        int(round(0.25 * (Z - 1))),
        int(round(0.50 * (Z - 1))),
        int(round(0.75 * (Z - 1))),
    ]
    slice_ids = [max(0, min(Z - 1, s)) for s in slice_ids]
    print(f"Case: {case_id} | volume shape: {img_np.shape} | slices: {slice_ids}", flush=True)

    out_dir = RUN_DIR / "viz_3d"
    out_dir.mkdir(parents=True, exist_ok=True)

    for s in slice_ids:
        img2d = img_np[s]     # [Y,X]
        lbl2d = lbl_np[s]     # [Y,X]
        pred2d = pred_np[s]   # [Y,X]

        img01 = _to_01(img2d)
        lbl_overlay = _bright_overlay(img01, lbl2d)
        pred_overlay = _bright_overlay(img01, pred2d)

        fig = plt.figure(figsize=(12, 10))

        ax = fig.add_subplot(2, 2, 1)
        ax.imshow(lbl2d, cmap="gray")
        ax.set_title(f"GT mask | z={s}")
        ax.axis("off")

        ax = fig.add_subplot(2, 2, 2)
        ax.imshow(pred2d, cmap="gray")
        ax.set_title(f"Pred mask (thr=0.5) | z={s}")
        ax.axis("off")

        ax = fig.add_subplot(2, 2, 3)
        ax.imshow(lbl_overlay, cmap="gray")
        ax.set_title("Image + GT overlay (brightened)")
        ax.axis("off")

        ax = fig.add_subplot(2, 2, 4)
        ax.imshow(pred_overlay, cmap="gray")
        ax.set_title("Image + Pred overlay (brightened)")
        ax.axis("off")

        fig.suptitle(f"{case_id} | slice z={s}", fontsize=14)
        fig.tight_layout()

        out_path = out_dir / f"{case_id}_z{s:03d}_4panel.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {out_path}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
