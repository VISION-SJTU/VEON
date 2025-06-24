import math

import fvcore.nn.weight_init as weight_init
import torch
from detectron2.layers import CNNBlockBase, Conv2d
from torch import nn
from torch.nn import functional as F
from typing import Tuple


class LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(
        self, input_dim, hidden_dim, output_dim, num_layers, affine_func=nn.Linear
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            affine_func(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x: torch.Tensor):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MLP_3D(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(
        self, input_dim, hidden_dim, output_dim, num_layers, affine_func=nn.Linear, pre_func=None,
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            affine_func(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.pre_layer = pre_func(input_dim, input_dim) if pre_func is not None else None

    def forward(self, x: torch.Tensor):
        if self.pre_layer is not None:
            x = self.pre_layer(x)
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class AddFusion(CNNBlockBase):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, 1)
        self.input_proj = nn.Sequential(
            LayerNorm(in_channels),
            Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
            ),
        )
        weight_init.c2_xavier_fill(self.input_proj[-1])

    def forward(self, x: torch.Tensor, y: torch.Tensor, spatial_shape: tuple):
        # x: [N,L,C] y: [N,C,H,W]
        y = (
            F.interpolate(
                self.input_proj(y.contiguous()),
                size=spatial_shape,
                mode="bilinear",
                align_corners=False,
            )
            .permute(0, 2, 3, 1)
            .reshape(x.shape)
        )
        x = x + y
        return x


def build_fusion_layer(fusion_type: str, in_channels: int, out_channels: int):
    if fusion_type == "add":
        return AddFusion(in_channels, out_channels)
    else:
        raise ValueError("Unknown fusion type: {}".format(fusion_type))


class AddFusionLift(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, out_channels):
        super().__init__()
        self.input_proj_1 = nn.Sequential(
            LayerNorm(in_channels_1),
            Conv2d(
                in_channels_1,
                out_channels,
                kernel_size=1,
            ),
        )
        weight_init.c2_xavier_fill(self.input_proj_1[-1])
        self.input_proj_2 = nn.Sequential(
            LayerNorm(in_channels_2),
            Conv2d(
                in_channels_2,
                out_channels,
                kernel_size=1,
            ),
        )
        self.relu = nn.ReLU(inplace=True)
        weight_init.c2_xavier_fill(self.input_proj_2[-1])


    def forward(self, x: torch.Tensor, y: torch.Tensor, spatial_shape: tuple):
        # x: [N,L,C]
        # y: [N,C,H,W]
        # if spatial_shape is None:
        #     N, C, H, W = y.shape
        #     ratio = round(math.sqrt(L // (H * W)))
        #     xH, xW = H * ratio, W * ratio
        #     spatial_shape = (xH, xW)

        x = self.input_proj_1(x)
        y = F.interpolate(
                self.input_proj_2(y.contiguous()),
                size=spatial_shape,
                mode="bilinear",
                align_corners=False)

        return self.relu(x + y)


class CatFusionLift(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, out_channels):
        super().__init__()
        out_channels_p1 = out_channels // 4
        out_channels_p2 = out_channels - out_channels_p1
        self.input_proj_1 = nn.Sequential(
            LayerNorm(in_channels_1 + in_channels_2),
            Conv2d(
                in_channels_1 + in_channels_2,
                out_channels_p1,
                kernel_size=1,
            ),
        )
        weight_init.c2_xavier_fill(self.input_proj_1[-1])
        self.input_proj_2 = nn.Sequential(
            LayerNorm(in_channels_2),
            Conv2d(
                in_channels_2,
                out_channels_p2,
                kernel_size=1,
            ),
        )
        self.relu = nn.ReLU(inplace=True)
        weight_init.c2_xavier_fill(self.input_proj_2[-1])


    def forward(self, x1: torch.Tensor, x2: torch.Tensor, spatial_shape: Tuple[int, int]):
        # x: [N,L,C]
        # y: [N,C,H,W]
        # if spatial_shape is None:
        #     N, C, H, W = y.shape
        #     ratio = round(math.sqrt(L // (H * W)))
        #     xH, xW = H * ratio, W * ratio
        #     spatial_shape = (xH, xW)
        if x2.shape[-2:] != spatial_shape:
            x2 = F.interpolate(x2.contiguous(), size=spatial_shape,
                    mode="bilinear", align_corners=False)
        if x1.shape[-2:] != spatial_shape:
            x1 = F.interpolate(x1.contiguous(), size=spatial_shape,
                    mode="bilinear", align_corners=False)

        y1 = self.input_proj_1(torch.cat([x1, x2], dim=1))
        y2 = self.input_proj_2(x2)
        y = torch.cat([y1, y2], dim=1)

        return self.relu(y)


def build_fusion_layer_lift(fusion_type: str, in_channels_1: int, in_channels_2: int, out_channels: int):
    if fusion_type == "add_fusion":
        return AddFusionLift(in_channels_1, in_channels_2, out_channels)
    elif fusion_type == "cat_fusion":
        return CatFusionLift(in_channels_1, in_channels_2, out_channels)
    else:
        raise ValueError("Unknown fusion type: {}".format(fusion_type))
