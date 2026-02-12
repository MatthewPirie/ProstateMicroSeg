# src/models/monai_unet_2d.py

from __future__ import annotations
from typing import Dict, Tuple

from monai.networks.nets import UNet


_UNET_2D_VARIANTS: Dict[str, Dict] = {
    "base": {
        "channels": (32, 64, 128, 256, 512, 512, 512, 512),
        "strides":  (2, 2, 2, 2, 2, 2, 2),
        "norm": "INSTANCE",
        "num_res_units": 0,
    },
    "small": {
        "channels": (32, 64, 128, 256, 512),
        "strides":  (2, 2, 2, 2),
        "norm": "INSTANCE",
        "num_res_units": 0,
    }
}

def build_monai_unet_2d(
    in_channels: int = 1,
    out_channels: int = 1,
    variant: str = "base",
):
    if variant not in _UNET_2D_VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Options: {list(_UNET_2D_VARIANTS.keys())}")

    cfg = _UNET_2D_VARIANTS[variant]

    model = UNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=cfg["channels"],
        strides=cfg["strides"],
        num_res_units=cfg["num_res_units"],
        norm=cfg["norm"],
    )

    # Return the model + a small metadata dict for printing/saving
    meta = {
        "model_name": "monai_unet_2d",
        "model_variant": variant,
        "channels": cfg["channels"],
        "strides": cfg["strides"],
        "norm": cfg["norm"],
        "num_res_units": cfg["num_res_units"],
    }
    return model, meta
