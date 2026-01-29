import torch
from monai.networks.nets import UNet

def build_monai_unet_2d(
    in_channels: int = 1,
    out_channels: int = 1,
):
    model = UNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=0,
        norm="INSTANCE",
    )
    return model
