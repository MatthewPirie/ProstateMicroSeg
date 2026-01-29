from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import SimpleITK as sitk
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def read_case_ids(txt_path: Path) -> List[str]:
    case_ids = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                case_ids.append(line)
    return case_ids


def center_crop_or_pad_2d(arr: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """
    Center-crop if arr is larger than target, pad (symmetric) if smaller.
    arr: (H, W)
    """
    assert arr.ndim == 2, f"Expected 2D array, got {arr.ndim}D"
    th, tw = target_hw
    h, w = arr.shape

    # Center crop (if needed)
    if h > th:
        top = (h - th) // 2
        arr = arr[top:top + th, :]
    if w > tw:
        left = (w - tw) // 2
        arr = arr[:, left:left + tw]

    # Pad (if needed)
    h2, w2 = arr.shape
    pad_h = max(th - h2, 0)
    pad_w = max(tw - w2, 0)

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if pad_h > 0 or pad_w > 0:
        arr = np.pad(
            arr,
            pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0,
        )

    return arr


def zscore_normalize(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    m = float(arr.mean())
    s = float(arr.std())
    if s < eps:
        return arr - m
    return (arr - m) / s


@dataclass(frozen=True)
class SplitSpec:
    name: str                 # "train" | "val" | "test"
    case_ids: List[str]
    images_subdir: str        # "imagesTr" or "imagesTs"
    labels_subdir: str        # "labelsTr" or "labelsTs"


class MicroUS2DSliceDataset(Dataset):
    """
    Returns one 2D slice (image, label) per item.

    - Loads NIfTI volumes from nnU-Net style folders.
    - Builds an index of (case_id, slice_idx) across all cases in a split.
    - Optionally transposes H/W to match nnU-Net's 2D orientation preferences.
    - Applies center crop/pad to a fixed size and z-score normalization.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        splits_dir: str | Path,
        split: str,  # "train" | "val" | "test"
        target_hw: Tuple[int, int] = (896, 1408),
        transpose_hw: bool = True,
        only_foreground_slices: bool = False,
    ):
        self.dataset_root = Path(dataset_root)
        self.splits_dir = Path(splits_dir)
        self.split = split
        self.target_hw = target_hw
        self.transpose_hw = transpose_hw
        self.only_foreground_slices = only_foreground_slices

        split_spec = self._make_split_spec(split)
        self.case_ids = split_spec.case_ids
        self.images_dir = self.dataset_root / split_spec.images_subdir
        self.labels_dir = self.dataset_root / split_spec.labels_subdir

        # Simple cache: keep most recently used case in memory
        self._cache_case_id: Optional[str] = None
        self._cache_img: Optional[np.ndarray] = None
        self._cache_lbl: Optional[np.ndarray] = None
        # Build slice index: list of (case_id, slice_idx)

        self.index: List[Tuple[str, int]] = self._build_index()


    def _make_split_spec(self, split: str) -> SplitSpec:
        split = split.lower()
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")

        case_ids = read_case_ids(self.splits_dir / f"{split}.txt")

        if split == "test":
            return SplitSpec(name=split, case_ids=case_ids, images_subdir="imagesTs", labels_subdir="labelsTs")
        else:
            return SplitSpec(name=split, case_ids=case_ids, images_subdir="imagesTr", labels_subdir="labelsTr")

    def _case_paths(self, case_id: str) -> Tuple[Path, Path]:
        # processed format: <images_dir>/<case_id>.npy and <labels_dir>/<case_id>.npy
        img_path = self.images_dir / f"{case_id}.npy"
        lbl_path = self.labels_dir / f"{case_id}.npy"
        if not img_path.exists():
            raise FileNotFoundError(f"Missing image: {img_path}")
        if not lbl_path.exists():
            raise FileNotFoundError(f"Missing label: {lbl_path}")
        return img_path, lbl_path

    def _load_case(self, case_id: str) -> Tuple[np.ndarray, np.ndarray]:
        if self._cache_case_id == case_id and self._cache_img is not None and self._cache_lbl is not None:
            return self._cache_img, self._cache_lbl

        # Now dataset_root contains imagesTr/labelsTr/imagesTs/labelsTs with .npy files
        img_path = self.images_dir / f"{case_id}.npy"
        lbl_path = self.labels_dir / f"{case_id}.npy"
        if not img_path.exists():
            raise FileNotFoundError(f"Missing image: {img_path}")
        if not lbl_path.exists():
            raise FileNotFoundError(f"Missing label: {lbl_path}")

        # Memory-mapped load for fast random access
        # Arrays are stored as (S, H, W)
        img = np.load(img_path, mmap_mode="r")
        lbl = np.load(lbl_path, mmap_mode="r")

        # Cache
        self._cache_case_id = case_id
        self._cache_img = img
        self._cache_lbl = lbl
        return img, lbl


    def _build_index(self) -> List[Tuple[str, int]]:
        idx: List[Tuple[str, int]] = []
        for cid in self.case_ids:
            img_path, _ = self._case_paths(cid)

            # .npy is stored as (S, H, W)
            img = np.load(img_path, mmap_mode="r")
            num_slices = img.shape[0]

            for s in range(num_slices):
                idx.append((cid, s))

        if len(idx) == 0:
            raise RuntimeError("Index is empty. Check split files and data paths.")
        return idx

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        case_id, s = self.index[i]
        img3d, lbl3d = self._load_case(case_id)

        img2d = img3d[s, :, :]
        lbl2d = lbl3d[s, :, :]

        # Optional transpose to switch (H, W) <-> (W, H)
        # This is useful if you want to align with nnU-Net's internal orientation.
        if self.transpose_hw:
            img2d = img2d.T
            lbl2d = lbl2d.T

        # Crop/pad to target
        img2d = center_crop_or_pad_2d(img2d, self.target_hw)
        lbl2d = center_crop_or_pad_2d(lbl2d, self.target_hw)

        # Z-score normalize image (not label)
        img2d = zscore_normalize(img2d)

        # Ensure label is binary 0/1
        lbl2d = (lbl2d > 0.5).astype(np.float32)

        # Convert to torch tensors
        # Image: (1, H, W), Label: (1, H, W)
        img_t = torch.from_numpy(img2d).unsqueeze(0)  # float32
        lbl_t = torch.from_numpy(lbl2d).unsqueeze(0)  # float32

        return {
            "image": img_t,
            "label": lbl_t,
            "case_id": case_id,
            "slice_idx": torch.tensor(s, dtype=torch.int64),
        }
