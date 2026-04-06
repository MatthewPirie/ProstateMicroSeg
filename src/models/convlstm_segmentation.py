# src/models/segmentation_convlstm.py

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.convlstm import ConvLSTM


_MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
    "small": {
        "encoder_channels": (32, 64, 128),
        "convlstm_hidden_dim": 128,
        "convlstm_num_layers": 1,
        "convlstm_kernel_size": (3, 3),
    },
    "base": {
        "encoder_channels": (32, 64, 128, 256),
        "convlstm_hidden_dim": 256,
        "convlstm_num_layers": 1,
        "convlstm_kernel_size": (3, 3),
    },
    "large": {
        "encoder_channels": (32, 64, 128, 256, 512),
        "convlstm_hidden_dim": 512,
        "convlstm_num_layers": 1,
        "convlstm_kernel_size": (3, 3),
    },
}


def _reshape_btchw_to_bczhw(x: torch.Tensor) -> torch.Tensor:
    # (B, T, C, H, W) -> (B, C, T, H, W)
    return x.permute(0, 2, 1, 3, 4).contiguous()


class ConvBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock2D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBlock2D(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class SegmentationConvLSTM(nn.Module):
    """
    Input:
        (B, C, Z, H, W)

    Output:
        (B, out_channels, Z, H, W)

    Structure:
        2D encoder on each frame
        -> ConvLSTM at bottleneck across time
        -> 2D decoder on each frame
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        encoder_channels: Sequence[int] = (32, 64, 128, 256),
        convlstm_hidden_dim: int = 256,
        convlstm_num_layers: int = 1,
        convlstm_kernel_size: Tuple[int, int] = (3, 3),
    ) -> None:
        super().__init__()

        if len(encoder_channels) < 2:
            raise ValueError("encoder_channels must have at least 2 entries")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.encoder_channels = tuple(int(c) for c in encoder_channels)
        self.convlstm_hidden_dim = int(convlstm_hidden_dim)

        # -------------------------
        # Encoder
        # -------------------------
        self.enc_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        prev_ch = self.in_channels
        for ch in self.encoder_channels:
            self.enc_blocks.append(ConvBlock2D(prev_ch, ch))
            prev_ch = ch

        # -------------------------
        # Temporal bottleneck
        # -------------------------
        self.temporal = ConvLSTM(
            input_dim=self.encoder_channels[-1],
            hidden_dim=self.convlstm_hidden_dim,
            kernel_size=convlstm_kernel_size,
            num_layers=convlstm_num_layers,
            batch_first=True,
            bias=True,
            return_all_layers=False,
        )

        # -------------------------
        # Decoder
        # Mirror encoder in reverse, excluding bottleneck
        # Example:
        # encoder: [32, 64, 128, 256]
        # skips:   [32, 64, 128]
        # decoder: 256/hidden -> 128 -> 64 -> 32
        # -------------------------
        skip_channels = list(self.encoder_channels[:-1])
        skip_channels_rev = list(reversed(skip_channels))

        self.dec_blocks = nn.ModuleList()

        dec_in = self.convlstm_hidden_dim
        for skip_ch in skip_channels_rev:
            self.dec_blocks.append(
                UpBlock2D(
                    in_channels=dec_in,
                    skip_channels=skip_ch,
                    out_channels=skip_ch,
                )
            )
            dec_in = skip_ch

        self.head = nn.Conv2d(dec_in, self.out_channels, kernel_size=1)

    def _encode_frames(
        self,
        x: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        x: (B, C, Z, H, W)

        Returns:
            skips: list of skip tensors, each (B, Z, C_i, H_i, W_i)
            bottleneck: (B, Z, C_last, H_last, W_last)
        """
        b, c, z, h, w = x.shape

        # Apply 2D encoder frame-by-frame using batch flattening
        x2d = x.permute(0, 2, 1, 3, 4).reshape(b * z, c, h, w)

        skips: List[torch.Tensor] = []

        for level_idx, block in enumerate(self.enc_blocks):
            x2d = block(x2d)

            ch = x2d.shape[1]
            hh = x2d.shape[2]
            ww = x2d.shape[3]

            x_level = x2d.view(b, z, ch, hh, ww)

            # store all except final bottleneck stage as skips
            if level_idx < len(self.enc_blocks) - 1:
                skips.append(x_level)
                x2d = self.pool(x2d)

        bottleneck = x2d.view(b, z, x2d.shape[1], x2d.shape[2], x2d.shape[3])
        return skips, bottleneck

    def _decode_frames(
        self,
        temporal_feats: torch.Tensor,
        skips: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        temporal_feats: (B, Z, C, H, W)
        skips: list of (B, Z, C_i, H_i, W_i), shallow -> deep
        """
        b, z, c, h, w = temporal_feats.shape
        x = temporal_feats.reshape(b * z, c, h, w)

        skips_rev = list(reversed(skips))

        for dec_block, skip in zip(self.dec_blocks, skips_rev):
            skip2d = skip.reshape(
                b * z,
                skip.shape[2],
                skip.shape[3],
                skip.shape[4],
            )
            x = dec_block(x, skip2d)

        x = self.head(x)  # (B*Z, out_channels, H, W)

        x = x.view(b, z, x.shape[1], x.shape[2], x.shape[3])  # (B, Z, C, H, W)
        x = _reshape_btchw_to_bczhw(x)  # (B, C, Z, H, W)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                f"Expected input shape (B, C, Z, H, W), got {tuple(x.shape)}"
            )

        skips, bottleneck = self._encode_frames(x)

        temporal_outputs, _ = self.temporal(bottleneck)
        temporal_feats = temporal_outputs[0]  # (B, Z, hidden_dim, H, W)

        logits = self._decode_frames(temporal_feats, skips)
        return logits


def build_segmentation_convlstm(
    in_channels: int = 1,
    out_channels: int = 1,
    variant: str = "base",
):
    if variant not in _MODEL_VARIANTS:
        raise ValueError(
            f"Unknown variant '{variant}'. Available variants: {list(_MODEL_VARIANTS.keys())}"
        )

    cfg = dict(_MODEL_VARIANTS[variant])

    model = SegmentationConvLSTM(
        in_channels=in_channels,
        out_channels=out_channels,
        **cfg,
    )

    meta = {
        "name": "SegmentationConvLSTM",
        "variant": variant,
        "in_channels": in_channels,
        "out_channels": out_channels,
        **cfg,
    }

    return model, meta