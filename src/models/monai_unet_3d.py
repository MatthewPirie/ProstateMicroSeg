# src/models/monai_unet_3d.py

from __future__ import annotations
from typing import Dict, Any, Tuple, List, Optional

import torch

# We prefer DynUNet because it lets us specify per-stage (anisotropic) kernel sizes + strides,
# which matches nnU-Net's 3D_fullres plan much better than monai.networks.nets.UNet.
try:
    from monai.networks.nets import DynUNet
    _HAS_DYNUNET = True
except Exception:
    DynUNet = None
    _HAS_DYNUNET = False

from monai.networks.nets import UNet


_UNET_3D_VARIANTS: Dict[str, Dict[str, Any]] = {
    # Matches your nnU-Net 3d_fullres plan *as closely as MONAI allows*
    # Plan:
    #   n_stages = 7
    #   features_per_stage = [32,64,128,256,320,320,320]
    #   kernel_sizes: [(1,3,3),(1,3,3),(3,3,3),(3,3,3),(3,3,3),(3,3,3),(3,3,3)]
    #   strides:      [(1,1,1),(1,2,2),(1,2,2),(2,2,2),(1,2,2),(1,2,2),(1,2,2)]
    "nnunet_fullres": {
        "filters": (32, 64, 128, 256, 320, 320, 320),
        "kernel_sizes": (
            (1, 3, 3),
            (1, 3, 3),
            (3, 3, 3),
            (3, 3, 3),
            (3, 3, 3),
            (3, 3, 3),
            (3, 3, 3),
        ),
        "strides": (
            (1, 1, 1),
            (1, 2, 2),
            (1, 2, 2),
            (2, 2, 2),
            (1, 2, 2),
            (1, 2, 2),
            (1, 2, 2),
        ),
        "norm": "INSTANCE",
        "num_res_units": 0,  # nnU-Net PlainConvUNet is not residual
        "deep_supervision": False,  # keep off for now (we can add later)
    },

    # Smaller sanity-check variant (faster, less memory)
    "small": {
        "filters": (32, 64, 128, 256, 320),
        "kernel_sizes": (
            (1, 3, 3),
            (1, 3, 3),
            (3, 3, 3),
            (3, 3, 3),
            (3, 3, 3),
        ),
        "strides": (
            (1, 1, 1),
            (1, 2, 2),
            (2, 2, 2),
            (1, 2, 2),
            (1, 2, 2),
        ),
        "norm": "INSTANCE",
        "num_res_units": 0,
        "deep_supervision": False,
    },
}


def build_monai_unet_3d(
    in_channels: int = 1,
    out_channels: int = 1,
    variant: str = "nnunet_fullres",
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Builds a 3D U-Net that approximates nnU-Net's 3D_fullres architecture.

    Preferred: DynUNet (supports per-stage kernel sizes + strides).
    Fallback: MONAI UNet (cannot exactly match anisotropic kernels per stage).
    """
    if variant not in _UNET_3D_VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Options: {list(_UNET_3D_VARIANTS.keys())}")

    cfg = _UNET_3D_VARIANTS[variant]

    if _HAS_DYNUNET:
        # DynUNet expects:
        #  - kernel_size: list/tuple of length n_stages
        #  - strides: list/tuple of length n_stages (first should be (1,1,1))
        #  - upsample_kernel_size: usually strides[1:] (one per upsample)
        model = DynUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=list(cfg["kernel_sizes"]),
            strides=list(cfg["strides"]),
            upsample_kernel_size=list(cfg["strides"][1:]),
            filters=list(cfg["filters"]),
            norm_name=cfg["norm"],
            deep_supervision=bool(cfg.get("deep_supervision", False)),
            # Leave residual blocks off to mirror nnU-Net PlainConvUNet
            res_block=False,
        )
        model_name = "monai_dynunet_3d"
    else:
        # Fallback: cannot match per-stage kernel anisotropy.
        # We approximate downsampling strategy by using anisotropic strides if your MONAI version supports it.
        # Some MONAI versions expect strides as ints; if yours errors, use a pure isotropic int list instead.
        try:
            model = UNet(
                spatial_dims=3,
                in_channels=in_channels,
                out_channels=out_channels,
                channels=cfg["filters"],
                strides=cfg["strides"][1:],  # UNet expects downsample strides per level (no initial (1,1,1))
                num_res_units=cfg["num_res_units"],
                norm=cfg["norm"],
            )
        except Exception:
            # Last resort: isotropic strides (won't match nnU-Net well, but will run)
            isotropic_strides = (2,) * (len(cfg["filters"]) - 1)
            model = UNet(
                spatial_dims=3,
                in_channels=in_channels,
                out_channels=out_channels,
                channels=cfg["filters"],
                strides=isotropic_strides,
                num_res_units=cfg["num_res_units"],
                norm=cfg["norm"],
            )
        model_name = "monai_unet_3d"

    meta: Dict[str, Any] = {
        "model_name": model_name,
        "model_variant": variant,
        "filters": tuple(cfg["filters"]),
        "kernel_sizes": tuple(cfg["kernel_sizes"]),
        "strides": tuple(cfg["strides"]),
        "norm": cfg["norm"],
        "num_res_units": cfg["num_res_units"],
        "deep_supervision": bool(cfg.get("deep_supervision", False)),
        "uses_dynunet": _HAS_DYNUNET,
    }
    return model, meta
