import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from src.data.dataset_2d import MicroUS2DSliceDataset
from torch.utils.data import DataLoader
from src.models.monai_unet_2d import build_monai_unet_2d
import torch.nn as nn

DATA_ROOT = Path("/home/pirie03/projects/aip-medilab/pirie03/ProstateMicroSeg/dataset/raw/Dataset120_MicroUSProstate")
SPLITS_DIR = Path("/home/pirie03/projects/aip-medilab/pirie03/ProstateMicroSeg/dataset/splits")

ds = MicroUS2DSliceDataset(
    dataset_root=DATA_ROOT,
    splits_dir=SPLITS_DIR,
    split="train",
    target_hw=(896, 1408),
    transpose_hw=True,
    only_foreground_slices=False,
)

print("len(ds):", len(ds))

sample = ds[0]
img = sample["image"]
lbl = sample["label"]

print("case_id:", sample["case_id"])
print("slice_idx:", int(sample["slice_idx"]))
print("image shape:", tuple(img.shape), "dtype:", img.dtype, "min/max:", float(img.min()), float(img.max()))
print("label shape:", tuple(lbl.shape), "dtype:", lbl.dtype, "unique:", torch.unique(lbl))

loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)

batch = next(iter(loader))
imgs = batch["image"]
lbls = batch["label"]

print("BATCH image shape:", tuple(imgs.shape))
print("BATCH label shape:", tuple(lbls.shape))
print("BATCH label unique:", torch.unique(lbls))

model = build_monai_unet_2d()
model.eval()

with torch.no_grad():
    out = model(imgs)  # imgs from your batch
print("MODEL output shape:", tuple(out.shape), "dtype:", out.dtype)


criterion = nn.BCEWithLogitsLoss()
loss = criterion(out, lbls)
print("LOSS:", float(loss))

model.train()  # enable grads
out = model(imgs)  # forward with grads
loss = criterion(out, lbls)

model.zero_grad()
loss.backward()

print("BACKWARD OK. grad mean (first param):", float(next(model.parameters()).grad.abs().mean()))


