# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import torch
import torch.nn as nn
import numpy as np
from ..midas.dpt_depth import DPTDepthModel

def get_activation(name, bank):
    def hook(model, input, output):
        bank[name] = output
    return hook

class MidasCore(nn.Module):
    def __init__(self, midas, trainable=False, fetch_features=True,
                 layer_names=('out_conv', 'l4_rn', 'r4', 'r3', 'r2', 'r1'),
                 freeze_bn=False, keep_aspect_ratio=True, img_size=384, **kwargs):
        """Midas Base model used for multi-scale feature extraction.

        Args:
            midas (torch.nn.Module): Midas model.
            trainable (bool, optional): Train midas model. Defaults to False.
            fetch_features (bool, optional): Extract multi-scale features. Defaults to True.
            layer_names (tuple, optional): Layers used for feature extraction. Order = (head output features, last layer features, ...decoder features). Defaults to ('out_conv', 'l4_rn', 'r4', 'r3', 'r2', 'r1').
            freeze_bn (bool, optional): Freeze BatchNorm. Generally results in better finetuning performance. Defaults to False.
            keep_aspect_ratio (bool, optional): Keep the aspect ratio of input images while resizing. Defaults to True.
            img_size (int, tuple, optional): Input resolution. Defaults to 384.
        """
        super().__init__()
        self.core = midas
        self.output_channels = None
        self.core_out = {}
        self.trainable = trainable
        self.fetch_features = fetch_features
        # midas.scratch.output_conv = nn.Identity()
        self.handles = []
        # self.layer_names = ['out_conv','l4_rn', 'r4', 'r3', 'r2', 'r1']
        self.layer_names = layer_names

        self.set_trainable(trainable)
        self.set_fetch_features(fetch_features)

        if freeze_bn:
            self.freeze_bn()

    def set_trainable(self, trainable):
        self.trainable = trainable
        if trainable:
            self.unfreeze()
        else:
            self.freeze()
        return self

    def set_fetch_features(self, fetch_features):
        self.fetch_features = fetch_features
        if fetch_features:
            if len(self.handles) == 0:
                self.attach_hooks(self.core)
        else:
            self.remove_hooks()
        return self

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.trainable = False
        return self

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True
        self.trainable = True
        return self

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def forward(self, x, denorm=False, return_rel_depth=False):

        with torch.set_grad_enabled(self.trainable):

            # print("Input size to Midascore", x.shape)
            rel_depth = self.core(x)
            # print("Output from midas shape", rel_depth.shape)
            if not self.fetch_features:
                return rel_depth
        out = [self.core_out[k].contiguous() for k in self.layer_names]

        if return_rel_depth:
            return rel_depth, out
        return out

    def get_rel_pos_params(self):
        for name, p in self.core.pretrained.named_parameters():
            if "relative_position" in name:
                yield p

    def get_enc_params_except_rel_pos(self):
        for name, p in self.core.pretrained.named_parameters():
            if "relative_position" not in name:
                yield p

    def freeze_encoder(self, freeze_rel_pos=False):
        if freeze_rel_pos:
            for p in self.core.pretrained.parameters():
                p.requires_grad = False
        else:
            for p in self.get_enc_params_except_rel_pos():
                p.requires_grad = False
        return self

    def attach_hooks(self, midas):
        if len(self.handles) > 0:
            self.remove_hooks()
        if "out_conv" in self.layer_names:
            self.handles.append(list(midas.scratch.output_conv.children())[
                                3].register_forward_hook(get_activation("out_conv", self.core_out)))
        if "r4" in self.layer_names:
            self.handles.append(midas.scratch.refinenet4.register_forward_hook(
                get_activation("r4", self.core_out)))
        if "r3" in self.layer_names:
            self.handles.append(midas.scratch.refinenet3.register_forward_hook(
                get_activation("r3", self.core_out)))
        if "r2" in self.layer_names:
            self.handles.append(midas.scratch.refinenet2.register_forward_hook(
                get_activation("r2", self.core_out)))
        if "r1" in self.layer_names:
            self.handles.append(midas.scratch.refinenet1.register_forward_hook(
                get_activation("r1", self.core_out)))
        if "l4_rn" in self.layer_names:
            self.handles.append(midas.scratch.layer4_rn.register_forward_hook(
                get_activation("l4_rn", self.core_out)))

        return self

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        return self

    def __del__(self):
        self.remove_hooks()

    def set_output_channels(self, model_type):
        self.output_channels = MIDAS_SETTINGS[model_type]

    @staticmethod
    def build(midas_model_type="DPT_BEiT_L_384", train_midas=False, use_pretrained_midas=True,
              fetch_features=False, freeze_bn=True, force_keep_ar=False, force_reload=False,
              backbone="beitl16_384", non_negative=True, **kwargs):
        if midas_model_type not in MIDAS_SETTINGS:
            raise ValueError(
                f"Invalid model type: {midas_model_type}. Must be one of {list(MIDAS_SETTINGS.keys())}")
        if "img_size" in kwargs:
            kwargs = MidasCore.parse_img_size(kwargs)
        img_size = kwargs.pop("img_size", [384, 384])
        print("img_size", img_size)
        midas = DPTDepthModel(backbone=backbone, non_negative=non_negative)
        kwargs.update({'keep_aspect_ratio': force_keep_ar})
        midas_core = MidasCore(midas, trainable=train_midas, fetch_features=fetch_features,
                               freeze_bn=freeze_bn, img_size=img_size, **kwargs)
        midas_core.set_output_channels(midas_model_type)
        return midas_core

    @staticmethod
    def build_from_config(config):
        return MidasCore.build(**config)

    @staticmethod
    def parse_img_size(config):
        assert 'img_size' in config
        if isinstance(config['img_size'], str):
            assert "," in config['img_size'], "img_size should be a string with comma separated img_size=H,W"
            config['img_size'] = list(map(int, config['img_size'].split(",")))
            assert len(
                config['img_size']) == 2, "img_size should be a string with comma separated img_size=H,W"
        elif isinstance(config['img_size'], int):
            config['img_size'] = [config['img_size'], config['img_size']]
        else:
            assert isinstance(config['img_size'], list) and len(
                config['img_size']) == 2, "img_size should be a list of H,W"
        return config


nchannels2models = {
    tuple([256]*5): ["DPT_BEiT_L_384", "DPT_BEiT_L_512", "DPT_BEiT_B_384",
                     "DPT_SwinV2_L_384", "DPT_SwinV2_B_384", "DPT_SwinV2_T_256", "DPT_Large", "DPT_Hybrid"],
    (512, 256, 128, 64, 64): ["MiDaS_small"]
}

# Model name to number of output channels
MIDAS_SETTINGS = {m: k for k, v in nchannels2models.items()
                  for m in v
                  }
