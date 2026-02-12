# scripts/make_case_stats.py

from pathlib import Path
import json
import numpy as np

IMAGES_DIR = Path(
    "/home/pirie03/projects/aip-medilab/pirie03/ProstateMicroSeg/dataset/processed/Dataset120_MicroUSProstate/imagesTr"
)

OUT_PATH = IMAGES_DIR.parent / "case_stats.json"
EPS = 1e-8


def compute_stats(arr: np.ndarray):
    # ignore zero background
    mask = arr != 0
    if mask.sum() == 0:
        return {"mean": 0.0, "std": 1.0, "min": 0.0, "max": 0.0, "n": 0}

    vals = arr[mask].astype(np.float64, copy=False)

    mean = float(vals.mean())
    std = float(vals.std())
    if std < EPS:
        std = 1.0

    return {
        "mean": mean,
        "std": std,
        "min": float(vals.min()),
        "max": float(vals.max()),
        "n": int(vals.size),
    }


def main():
    npy_files = sorted(IMAGES_DIR.glob("*.npy"))
    if len(npy_files) == 0:
        raise RuntimeError(f"No .npy files found in {IMAGES_DIR}")

    stats = {}

    for p in npy_files:
        case_id = p.stem
        arr = np.load(p, mmap_mode="r")

        s = compute_stats(arr)
        stats[case_id] = s

        print(f"{case_id}: mean={s['mean']:.4f} std={s['std']:.4f}")

    with open(OUT_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved stats to: {OUT_PATH}")


if __name__ == "__main__":
    main()
