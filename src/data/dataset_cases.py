# src/data/dataset_cases.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import json
import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.normalization import zscore_with_stats

def read_case_ids(txt_path: Path) -> List[str]:
    case_ids: List[str] = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                case_ids.append(line)
    return case_ids


@dataclass(frozen=True)
class SplitSpec:
    name: str
    case_ids: List[str]
    images_subdir: str
    labels_subdir: str


class MicroUSCaseDataset(Dataset):
    """
    One item = one full 3D case (image volume, label volume).

    Intended use:
      - Validation / inference (case-level evaluation)
      - Can be used by BOTH 2D and 3D trainers:
          * 2D trainer: iterate slices internally, stack predictions
          * 3D trainer: sliding-window inference internally, stitch predictions

    File expectations:
      - images stored as .npy in:
            <dataset_root>/imagesTr/<case_id>.npy   (train/val)
            <dataset_root>/imagesTs/<case_id>.npy   (test)
      - labels stored as .npy in:
            <dataset_root>/labelsTr/<case_id>.npy   (train/val)
            <dataset_root>/labelsTs/<case_id>.npy   (test)

    Array expectations:
      - image: (Z, Y, X) or (S, H, W)
      - label: (Z, Y, X) with foreground > fg_threshold

    Returns:
      {
        "image": torch.FloatTensor [1, Z, Y, X],  # channel-first
        "label": torch.FloatTensor [1, Z, Y, X],  # binary {0,1}
        "case_id": str
      }
    """

    def __init__(
        self,
        dataset_root: str | Path,
        splits_dir: str | Path,
        split: str,  # "train" | "val" | "test"
        fg_threshold: float = 0.5,
        use_case_stats: bool = True,
        case_stats_path: str | Path | None = None,
        eps: float = 1e-8,
    ):
        self.dataset_root = Path(dataset_root)
        self.splits_dir = Path(splits_dir)
        self.split = split.lower()
        self.fg_threshold = float(fg_threshold)
        self.use_case_stats = bool(use_case_stats)
        self.eps = float(eps)

        split_spec = self._make_split_spec(self.split)
        self.case_ids = split_spec.case_ids
        self.images_dir = self.dataset_root / split_spec.images_subdir
        self.labels_dir = self.dataset_root / split_spec.labels_subdir

        # Index: one item per case
        self.index: List[str] = list(self.case_ids)
        if len(self.index) == 0:
            raise RuntimeError("Index is empty. Check split files and data paths.")

        # Optional: load precomputed mean/std per case
        self.case_stats: Dict[str, Dict[str, float]] = {}
        if self.use_case_stats:
            if case_stats_path is None:
                raise ValueError("use_case_stats=True but case_stats_path was not provided.")
            case_stats_path = Path(case_stats_path)
            if not case_stats_path.exists():
                raise FileNotFoundError(f"case_stats_path not found: {case_stats_path}")
            with open(case_stats_path, "r") as f:
                self.case_stats = json.load(f)

            # Basic sanity: must have keys for all cases in this split
            missing = [cid for cid in self.index if cid not in self.case_stats]
            if len(missing) > 0:
                ex = ", ".join(missing[:5])
                raise RuntimeError(
                    f"case_stats missing {len(missing)} case_ids for split='{self.split}'. "
                    f"Examples: {ex}"
                )

        # Cache: keep most recently used case in memory (memmap-backed)
        self._cache_case_id: Optional[str] = None
        self._cache_img: Optional[np.ndarray] = None
        self._cache_lbl: Optional[np.ndarray] = None

        self._printed_shape = False

    def _make_split_spec(self, split: str) -> SplitSpec:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")

        case_ids = read_case_ids(self.splits_dir / f"{split}.txt")
        if split == "test":
            return SplitSpec(name=split, case_ids=case_ids, images_subdir="imagesTs", labels_subdir="labelsTs")
        return SplitSpec(name=split, case_ids=case_ids, images_subdir="imagesTr", labels_subdir="labelsTr")

    def _case_paths(self, case_id: str) -> Tuple[Path, Path]:
        img_path = self.images_dir / f"{case_id}.npy"
        lbl_path = self.labels_dir / f"{case_id}.npy"
        if not img_path.exists():
            raise FileNotFoundError(f"Missing image: {img_path}")
        if not lbl_path.exists():
            raise FileNotFoundError(f"Missing label: {lbl_path}")
        return img_path, lbl_path

    def _load_case(self, case_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns memmap-backed arrays (cheap to load repeatedly).
        """
        if self._cache_case_id == case_id and self._cache_img is not None and self._cache_lbl is not None:
            return self._cache_img, self._cache_lbl

        img_path, lbl_path = self._case_paths(case_id)

        img = np.load(img_path, mmap_mode="r")  # (Z,Y,X)
        lbl = np.load(lbl_path, mmap_mode="r")  # (Z,Y,X)

        self._cache_case_id = case_id
        self._cache_img = img
        self._cache_lbl = lbl
        return img, lbl

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        case_id = self.index[int(i)]
        img3d, lbl3d = self._load_case(case_id)

        # Convert memmap to ndarray view (still cheap)
        img_np = np.asarray(img3d)
        lbl_np = np.asarray(lbl3d)

        # Normalize image
        if self.use_case_stats:
            stats = self.case_stats[case_id]
            mean = float(stats["mean"])
            std = float(stats["std"])
            img_np = zscore_with_stats(img_np, mean, std)
        else:
            mean = float(img_np.mean())
            std = float(img_np.std())
            img_np = zscore_with_stats(img_np, mean, std)

        # Binarize label
        lbl_np = (lbl_np > self.fg_threshold).astype(np.float32)

        # Convert to torch tensors, add channel dim: [C, Z, Y, X]
        img_t = torch.from_numpy(img_np).unsqueeze(0).float()
        lbl_t = torch.from_numpy(lbl_np).unsqueeze(0).float()

        return {
            "image": img_t,
            "label": lbl_t,
            "case_id": case_id,
        }
