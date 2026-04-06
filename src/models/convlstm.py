# src/models/convlstm.py

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


def _to_list(param, num_layers: int):
    if isinstance(param, list):
        return param
    return [param] * num_layers


class ConvLSTMCell(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: Tuple[int, int] = (3, 3),
        bias: bool = True,
    ) -> None:
        super().__init__()

        if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
            raise ValueError("kernel_size must be a tuple of length 2")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.kernel_size = tuple(int(k) for k in kernel_size)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        self.bias = bool(bias)

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(
        self,
        input_tensor: torch.Tensor,
        cur_state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)
        gates = self.conv(combined)

        i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(
        self,
        batch_size: int,
        image_size: Tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        height, width = image_size
        h = torch.zeros(
            batch_size,
            self.hidden_dim,
            height,
            width,
            device=device,
            dtype=dtype,
        )
        c = torch.zeros(
            batch_size,
            self.hidden_dim,
            height,
            width,
            device=device,
            dtype=dtype,
        )
        return h, c


class ConvLSTM(nn.Module):
    """
    Input:
        - batch_first=True:  (B, T, C, H, W)
        - batch_first=False: (T, B, C, H, W)

    Output:
        - layer_output_list: list of tensors, each shaped (B, T, C, H, W)
        - last_state_list:   list of (h, c) tuples
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | Sequence[int],
        kernel_size: Tuple[int, int] | Sequence[Tuple[int, int]],
        num_layers: int = 1,
        batch_first: bool = True,
        bias: bool = True,
        return_all_layers: bool = False,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        hidden_dim = _to_list(hidden_dim, num_layers)
        kernel_size = _to_list(kernel_size, num_layers)

        if len(hidden_dim) != num_layers or len(kernel_size) != num_layers:
            raise ValueError("hidden_dim and kernel_size must match num_layers")

        self.input_dim = int(input_dim)
        self.hidden_dim = [int(h) for h in hidden_dim]
        self.kernel_size = [tuple(k) for k in kernel_size]
        self.num_layers = int(num_layers)
        self.batch_first = bool(batch_first)
        self.bias = bool(bias)
        self.return_all_layers = bool(return_all_layers)

        cells: List[ConvLSTMCell] = []
        for layer_idx in range(self.num_layers):
            cur_input_dim = self.input_dim if layer_idx == 0 else self.hidden_dim[layer_idx - 1]
            cells.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[layer_idx],
                    kernel_size=self.kernel_size[layer_idx],
                    bias=self.bias,
                )
            )

        self.cell_list = nn.ModuleList(cells)

    def _init_hidden(
        self,
        batch_size: int,
        image_size: Tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        states = []
        for cell in self.cell_list:
            states.append(
                cell.init_hidden(
                    batch_size=batch_size,
                    image_size=image_size,
                    device=device,
                    dtype=dtype,
                )
            )
        return states

    def forward(
        self,
        input_tensor: torch.Tensor,
        hidden_state: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ):
        if input_tensor.ndim != 5:
            raise ValueError(
                f"ConvLSTM expected 5D input, got shape {tuple(input_tensor.shape)}"
            )

        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, t, _, h, w = input_tensor.shape

        if hidden_state is None:
            hidden_state = self._init_hidden(
                batch_size=b,
                image_size=(h, w),
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            )
        elif len(hidden_state) != self.num_layers:
            raise ValueError("hidden_state length must match num_layers")

        layer_output_list = []
        last_state_list = []

        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h_cur, c_cur = hidden_state[layer_idx]
            output_inner = []

            for time_idx in range(t):
                h_cur, c_cur = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, time_idx, :, :, :],
                    cur_state=(h_cur, c_cur),
                )
                output_inner.append(h_cur)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append((h_cur, c_cur))

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list