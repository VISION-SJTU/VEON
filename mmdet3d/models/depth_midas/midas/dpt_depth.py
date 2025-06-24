import torch
import torch.nn as nn

from mmdet3d.models.builder import BACKBONES, NECKS
from .base_model import BaseModel
from .blocks import (
    FeatureFusionBlock_custom,
    Interpolate,
    Projector,
    DepthHead,
    _make_encoder,
    forward_beit,
    forward_swin,
    # forward_levit,
    # forward_vit,
)
# from .backbones.levit import stem_b4_transpose
from timm.models.layers import get_act_layer


def _make_fusion_block(features, use_bn, size = None):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPT(BaseModel):
    def __init__(
        self,
        head,
        features=256,
        backbone="vitb_rn50_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        **kwargs
    ):

        super(DPT, self).__init__()

        self.channels_last = channels_last

        # For the Swin, Swin 2, LeViT and Next-ViT Transformers, the hierarchical architectures prevent setting the 
        # hooks freely. Instead, the hooks have to be chosen according to the ranges specified in the comments.
        hooks = {
            "beitl16_512": [5, 11, 17, 23],
            "beitl16_384": [5, 11, 17, 23],
            "beitb16_384": [2, 5, 8, 11],
            "swin2l24_384": [1, 1, 17, 1],  # Allowed ranges: [0, 1], [0,  1], [ 0, 17], [ 0,  1]
            "swin2b24_384": [1, 1, 17, 1],                  # [0, 1], [0,  1], [ 0, 17], [ 0,  1]
            "swin2t16_256": [1, 1, 5, 1],                   # [0, 1], [0,  1], [ 0,  5], [ 0,  1]
            "swinl12_384": [1, 1, 17, 1],                   # [0, 1], [0,  1], [ 0, 17], [ 0,  1]
            "next_vit_large_6m": [2, 6, 36, 39],            # [0, 2], [3,  6], [ 7, 36], [37, 39]
            "levit_384": [3, 11, 21],                       # [0, 3], [6, 11], [14, 21]
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
        }[backbone]

        if "next_vit" in backbone:
            in_features = {
                "next_vit_large_6m": [96, 256, 512, 1024],
            }[backbone]
        else:
            in_features = None

        # Instantiate backbone and reassemble blocks
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            False, # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks,
            use_readout=readout,
            in_features=in_features,
        )

        self.number_layers = len(hooks) if hooks is not None else 4
        size_refinenet3 = None
        self.scratch.stem_transpose = None

        if "beit" in backbone:
            self.forward_transformer = forward_beit
        elif "swin" in backbone:
            self.forward_transformer = forward_swin
        else:
            assert False, "Currently do not support backbones except beit and swin."
        # elif "next_vit" in backbone:
        #     from .backbones.next_vit import forward_next_vit
        #     self.forward_transformer = forward_next_vit
        # elif "levit" in backbone:
        #     self.forward_transformer = forward_levit
        #     size_refinenet3 = 7
        #     self.scratch.stem_transpose = stem_b4_transpose(256, 128, get_act_layer("hard_swish"))
        # else:
        #     self.forward_transformer = forward_vit

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn, size_refinenet3)
        if self.number_layers >= 4:
            self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = head
        self.reserve_middle_feat = False


    def forward(self, x):

        middle_feat = {}

        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layers = self.forward_transformer(self.pretrained, x)
        if self.number_layers == 3:
            layer_1, layer_2, layer_3 = layers
        else:
            layer_1, layer_2, layer_3, layer_4 = layers

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        if self.number_layers >= 4:
            layer_4_rn = self.scratch.layer4_rn(layer_4).contiguous()
            middle_feat.update({'l4_rn': layer_4_rn})

        if self.number_layers == 3:
            path_3 = self.scratch.refinenet3(layer_3_rn, size=layer_2_rn.shape[2:]).contiguous()
            middle_feat.update({'p3': path_3})
        else:
            path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:]).contiguous()
            path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:]).contiguous()
            middle_feat.update({'p3': path_3, 'p4': path_4})
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:]).contiguous()
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn).contiguous()
        middle_feat.update({'p1': path_1, 'p2': path_2})

        if self.scratch.stem_transpose is not None:
            path_1 = self.scratch.stem_transpose(path_1)

        out = self.scratch.output_conv(path_1).squeeze(dim=1).contiguous()
        middle_feat.update({'rd': out})

        return middle_feat if self.reserve_middle_feat else out


@NECKS.register_module()
class DPTDepthModel(DPT):
    def __init__(self, non_negative=True, **kwargs):
        features = kwargs["features"] if "features" in kwargs else 256
        head_features_1 = kwargs["head_features_1"] if "head_features_1" in kwargs else features
        head_features_2 = kwargs["head_features_2"] if "head_features_2" in kwargs else 32
        kwargs.pop("head_features_1", None)
        kwargs.pop("head_features_2", None)

        head = nn.Sequential(
            nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )

        super().__init__(head, **kwargs)

    def forward(self, x):
        return super().forward(x)


@NECKS.register_module()
class DPTDepthModelAdaptor(DPTDepthModel):
    def __init__(self, bin_embedding_dim=128, btlnck_features=256,
                 num_out_features=(256, 256, 256, 256), out_size=(48, 48),
                 loss_depth_weight=0.5, grid_config=None, **kwargs):
        super(DPTDepthModelAdaptor, self).__init__(**kwargs)
        self.loss_depth_weight = loss_depth_weight
        self.reserve_middle_feat = True
        self.out_size = out_size
        self.grid_config = grid_config

        self.seed_projector = Projector(
            btlnck_features, bin_embedding_dim, mlp_dim=bin_embedding_dim // 2)
        self.projectors = nn.ModuleList([
            Projector(num_out, bin_embedding_dim, mlp_dim=bin_embedding_dim // 2)
            for num_out in num_out_features
        ])
        self.fusers = nn.ModuleList([
            Projector(bin_embedding_dim, bin_embedding_dim, mlp_dim=bin_embedding_dim // 2)
            for _ in num_out_features
        ])
        depth_cfg = self.grid_config['depth']
        self.D = int((depth_cfg[1] - depth_cfg[0]) / depth_cfg[2])
        self.depth_head = DepthHead(bin_embedding_dim, self.D, mlp_dim=bin_embedding_dim)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        res = super().forward(x)
        accum_embed = self.seed_projector(res['l4_rn'])
        feats = [res['p4'], res['p3'], res['p2'], res['p1']]
        for i, (feat, projector, fuser) in enumerate(zip(feats, self.projectors, self.fusers)):
            cur = projector(feat)
            if cur.shape[-2:] != accum_embed.shape[-2:]:
                # accum_embed = self.up(accum_embed)
                accum_embed = nn.functional.interpolate(
                    accum_embed, cur.shape[-2:], mode='bilinear', align_corners=False)
            accum_embed = fuser(accum_embed + cur)

        rel_depth = res['rd'].unsqueeze(dim=1)
        rel_depth = 1.0 / (rel_depth + 1e-6)
        rel_depth = (rel_depth - rel_depth.min()) / \
                    (rel_depth.max() - rel_depth.min())
        # low_res_depth = torch.clip(, max=10000.0, min=0.0)
        abs_depth = self.depth_head(accum_embed, rel_depth)
        # abs_depth = torch.softmax(abs_depth, dim=1)
        return abs_depth
