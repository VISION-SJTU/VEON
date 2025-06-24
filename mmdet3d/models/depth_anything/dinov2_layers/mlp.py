# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/mlp.py


from typing import Callable, Optional

from torch import Tensor, nn
from .lora_layers import Linear as LoRALinear

class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
        lora_r: int = -1,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        if lora_r > 1:
            self.fc1 = LoRALinear(in_features, hidden_features, bias=bias, r=lora_r)
            self.fc2 = LoRALinear(hidden_features, out_features, bias=bias, r=lora_r)
        else:
            self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
            self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
