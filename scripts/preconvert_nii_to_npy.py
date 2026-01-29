# ProstateMicroSeg/scripts/preconvert_nii_to_npy.py
from pathlib import Path
import numpy as np
import SimpleITK as sitk

RAW_ROOT = Path("/home/pirie03/projects/aip-medilab/pirie03/ProstateMicroSeg/dataset/raw/Dataset120_MicroUSProstate")
OUT_ROOT = Path("/home/pirie03/projects/aip-medilab/pirie03/ProstateMicroSeg/dataset/processed/Dataset120_MicroUSProstate")

def convert_split(images_dir: Path, labels_dir: Path, out_images_dir: Path, out_labels_dir: Path) -> None:
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    # images are like: microUS_01_0000.nii.gz
    for img_nii in sorted(images_dir.glob("*_0000.nii.gz")):
        case_id = img_nii.name.replace("_0000.nii.gz", "")
        lbl_nii = labels_dir / f"{case_id}.nii.gz"

        printed = False

        if not lbl_nii.exists():
            print(f"[SKIP] missing label for {case_id}: {lbl_nii}")
            continue

        out_img = out_images_dir / f"{case_id}.npy"
        out_lbl = out_labels_dir / f"{case_id}.npy"

        if out_img.exists() and out_lbl.exists():
            print(f"[SKIP] {case_id} (exists)")
            continue

        print(f"[CONVERT] {case_id}")

        img = sitk.GetArrayFromImage(sitk.ReadImage(str(img_nii))).astype(np.float32)  # (S,H,W)
        lbl = sitk.GetArrayFromImage(sitk.ReadImage(str(lbl_nii))).astype(np.float32)  # (S,H,W)

        if not printed:
            print(f"[DEBUG] {case_id}")
            print(f"  image shape: {img.shape}  (expected: S,H,W)")
            print(f"  label shape: {lbl.shape}  (expected: S,H,W)")
            printed = True

        np.save(out_img, img)
        np.save(out_lbl, lbl)

def main():
    convert_split(
        images_dir=RAW_ROOT / "imagesTr",
        labels_dir=RAW_ROOT / "labelsTr",
        out_images_dir=OUT_ROOT / "imagesTr",
        out_labels_dir=OUT_ROOT / "labelsTr",
    )
    convert_split(
        images_dir=RAW_ROOT / "imagesTs",
        labels_dir=RAW_ROOT / "labelsTs",
        out_images_dir=OUT_ROOT / "imagesTs",
        out_labels_dir=OUT_ROOT / "labelsTs",
    )
    print("Done.")

if __name__ == "__main__":
    main()
