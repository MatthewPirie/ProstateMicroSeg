import sys
sys.path.append("/home/pirie03/projects/aip-medilab/pirie03/ProstateMicroSeg")

from pathlib import Path
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.models.monai_unet_3d import build_monai_unet_3d
from src.data.dataset_cases import MicroUSCaseDataset
from src.utils.metrics import dice_hard_from_logits

# -------------------------
# EDIT THESE
# -------------------------
RUN_DIR = Path("/home/pirie03/projects/aip-medilab/pirie03/ProstateMicroSeg/runs_3d/20260212_121431")
CKPT_PATH = RUN_DIR / "checkpoint_best.pt"
OUT_DIR = RUN_DIR / "viz_3d_patch"

# If you want to force a specific val case_id, set it here (else uses first val case)
CASE_ID_OVERRIDE = None  # e.g. "microUS_01"

# Patch size you trained on
ROI_ZYX = (14, 256, 448)

# Random patches per case
N_PATCHES = 3

# Repro seed for patch sampling
SEED = 0

# Threshold for binarizing prediction
THR = 0.5


def _pick_random_patch_start(vol_zyx, roi_zyx, rng):
    """
    Choose a random valid start index (z0,y0,x0) so that the crop fits in-bounds.
    Assumes vol dims >= roi dims in each axis. If not, we fall back to start=0
    and we will pad later.
    """
    Z, Y, X = vol_zyx
    rz, ry, rx = roi_zyx

    z0_max = max(Z - rz, 0)
    y0_max = max(Y - ry, 0)
    x0_max = max(X - rx, 0)

    z0 = int(rng.integers(0, z0_max + 1)) if z0_max > 0 else 0
    y0 = int(rng.integers(0, y0_max + 1)) if y0_max > 0 else 0
    x0 = int(rng.integers(0, x0_max + 1)) if x0_max > 0 else 0
    return z0, y0, x0


def _crop_or_pad_to_roi(img_zyx, lbl_zyx, start_zyx, roi_zyx):
    """
    Crop a ROI starting at start_zyx. If volume is smaller than ROI along any axis,
    pad with zeros to allow the ROI.
    Returns img_patch, lbl_patch shaped exactly roi_zyx.
    """
    img = img_zyx
    lbl = lbl_zyx
    Z, Y, X = img.shape
    rz, ry, rx = roi_zyx
    z0, y0, x0 = start_zyx

    # Pad if needed
    pad_z = max(rz - Z, 0)
    pad_y = max(ry - Y, 0)
    pad_x = max(rx - X, 0)

    if pad_z > 0 or pad_y > 0 or pad_x > 0:
        img = np.pad(img, ((0, pad_z), (0, pad_y), (0, pad_x)), mode="constant", constant_values=0.0)
        lbl = np.pad(lbl, ((0, pad_z), (0, pad_y), (0, pad_x)), mode="constant", constant_values=0.0)

    # Now crop
    img_patch = img[z0:z0+rz, y0:y0+ry, x0:x0+rx]
    lbl_patch = lbl[z0:z0+rz, y0:y0+ry, x0:x0+rx]

    # Safety: ensure exact shape
    assert img_patch.shape == roi_zyx, f"img_patch shape {img_patch.shape} != {roi_zyx}"
    assert lbl_patch.shape == roi_zyx, f"lbl_patch shape {lbl_patch.shape} != {roi_zyx}"
    return img_patch, lbl_patch


def _overlay(ax, base, mask, title):
    ax.imshow(base, cmap="gray")

    # Create yellow RGB mask
    yellow_mask = np.zeros((*mask.shape, 3))
    yellow_mask[..., 0] = 1.0  # Red
    yellow_mask[..., 1] = 1.0  # Green
    # Blue stays 0 → yellow

    ax.imshow(yellow_mask, alpha=0.35 * (mask > 0.5))
    ax.set_title(title)
    ax.axis("off")


def main():
    # -------------------------
    # Device
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # -------------------------
    # Load checkpoint + build model
    # -------------------------
    ckpt = torch.load(CKPT_PATH, map_location=device)

    variant = "nnunet_fullres"
    extra = ckpt.get("extra", {})
    if isinstance(extra, dict):
        args = extra.get("args", {})
        if isinstance(args, dict) and "model_variant" in args:
            variant = args["model_variant"]

    print("Model variant:", variant)

    model, _ = build_monai_unet_3d(in_channels=1, out_channels=1, variant=variant)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # -------------------------
    # Find dataset paths from ckpt (if present)
    # -------------------------
    data_root = None
    splits_dir = None
    use_case_stats = True
    case_stats_path = ""

    if isinstance(extra, dict):
        args = extra.get("args", {})
        if isinstance(args, dict):
            data_root = args.get("data_root", None)
            splits_dir = args.get("splits_dir", None)
            use_case_stats = bool(args.get("use_case_stats", True))
            case_stats_path = args.get("case_stats_path", "")

    if data_root is None or splits_dir is None:
        raise RuntimeError(
            "Could not auto-detect data_root/splits_dir from checkpoint. "
            "Your checkpoint extra['args'] is missing them."
        )

    # If case_stats_path is empty, MicroUSCaseDataset will require you to pass a real path
    # only if use_case_stats=True. So fix it here.
    if use_case_stats:
        if not case_stats_path:
            case_stats_path = str(Path(data_root) / "case_stats.json")

    # -------------------------
    # Load one full val case
    # -------------------------
    ds = MicroUSCaseDataset(
        dataset_root=data_root,
        splits_dir=splits_dir,
        split="val",
        use_case_stats=use_case_stats,
        case_stats_path=case_stats_path if use_case_stats else None,
        fg_threshold=0.5,
    )

    # pick case
    if CASE_ID_OVERRIDE is not None:
        if CASE_ID_OVERRIDE not in ds.index:
            raise RuntimeError(f"CASE_ID_OVERRIDE={CASE_ID_OVERRIDE} not in val split.")
        case_i = ds.index.index(CASE_ID_OVERRIDE)
    else:
        case_i = 0

    sample = ds[case_i]
    case_id = sample["case_id"]
    img = sample["image"].numpy()[0]  # [Z,Y,X]
    lbl = sample["label"].numpy()[0]  # [Z,Y,X] binary

    print("Case:", case_id, "img shape:", img.shape, "lbl shape:", lbl.shape)

    # -------------------------
    # Random patch sampling + inference
    # -------------------------
    rng = np.random.default_rng(SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Saving figures to:", OUT_DIR)

    for p in range(N_PATCHES):
        start_zyx = _pick_random_patch_start(img.shape, ROI_ZYX, rng)
        img_patch_np, lbl_patch_np = _crop_or_pad_to_roi(img, lbl, start_zyx, ROI_ZYX)

        # to torch: [1,1,Z,Y,X]
        img_t = torch.from_numpy(img_patch_np).unsqueeze(0).unsqueeze(0).float().to(device)
        lbl_t = torch.from_numpy(lbl_patch_np).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            logits = model(img_t)

        # dice on this patch
        d = float(dice_hard_from_logits(logits, lbl_t, threshold=THR).item())

        # get center z-slice for visualization
        zc = ROI_ZYX[0] // 2
        img2d = img_patch_np[zc, :, :]
        lbl2d = lbl_patch_np[zc, :, :]

        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()  # [Z,Y,X]
        pred2d = (prob[zc, :, :] > THR).astype(np.float32)

        # -------------------------
        # Plot 2x2
        # -------------------------
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle(f"{case_id} | patch {p+1}/{N_PATCHES} | start_zyx={start_zyx} | Dice={d:.3f}", fontsize=12)

        axes[0, 0].imshow(lbl2d, cmap="gray")
        axes[0, 0].set_title("Label (center slice)")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(pred2d, cmap="gray")
        axes[0, 1].set_title("Prediction (center slice)")
        axes[0, 1].axis("off")

        _overlay(axes[1, 0], img2d, lbl2d, "Image + Label overlay")
        _overlay(axes[1, 1], img2d, pred2d, "Image + Pred overlay")

        plt.tight_layout()

        # filename: case + patch index + dice
        out_path = OUT_DIR / f"{case_id}_patch{p+1:02d}_z{zc:02d}_dice{d:.3f}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)  # important to avoid memory buildup


if __name__ == "__main__":
    main()
