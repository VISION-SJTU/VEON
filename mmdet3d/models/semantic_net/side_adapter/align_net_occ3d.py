import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
from torch.utils.checkpoint import checkpoint
from mmcv.cnn.bricks.conv_module import ConvModule
from einops import rearrange
from mmdet3d.models.necks.view_transformer_raw import LSSViewTransformerRaw
from mmdet3d.utils.vis import vis_occ
from ..layers import build_fusion_layer_lift


class TemporalFusionDeformMiddle(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.t_deform = TemporalDeformable(channels=channels)
    
    def forward(self, mid_feat, occ_feat, prev_feat):
        deform_feat = self.t_deform(mid_feat, occ_feat)
        deform_feat_2 = self.t_deform(mid_feat, prev_feat)
        final_fused_feat = torch.cat([mid_feat, deform_feat, deform_feat_2], dim=1)
        return final_fused_feat


class TemporalFusionMultiFrameMiddle3x3Seq(nn.Module):
    def __init__(self, channels, seqs=2):
        super().__init__()
        self.t_fuse = nn.ModuleList([ConvModule(channels * 2, channels,
            kernel_size=3, stride=1, padding=1,
            bias=False, conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d')) for _ in range(seqs)])
    
    def forward(self, cur_occ_feat, prev_occ_feats):
        prev_feat = None
        idx = 0
        for cur_feat in prev_occ_feats[::-1]:
            if prev_feat is None:
                prev_feat = cur_feat
                continue
            fused_feat_raw = torch.cat([cur_feat, prev_feat], dim=1)
            prev_feat = self.t_fuse[idx](fused_feat_raw)
            idx += 1
        mid_feat_raw = torch.cat([cur_occ_feat, prev_feat], dim=1)
        ref_feat = self.t_fuse[idx](mid_feat_raw)        
        return ref_feat, prev_feat


class TemporalFusionMultiFrame(nn.Module):
    def __init__(self, channels, seqs=2):
        super().__init__()

        self.t_final = ConvModule(channels * 3, channels,
            kernel_size=3, stride=1, padding=1,
            bias=False, conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d'))
        
        # Before fusion layer
        self.before_fusion_layer = BeforeFusionLayer(channels)
        # Temporal fusion layer
        # self.t_fuse_mid = TemporalFusionMultiFrameMiddle(channels)
        self.t_fuse_mid = TemporalFusionMultiFrameMiddle3x3Seq(channels, seqs=seqs)
        # Deform layers
        self.deform_fusion_layer = TemporalFusionDeformMiddle(channels)
        
        
    def forward(self, cur_occ_feat, prev_occ_feats):
        before_fuse_feats = self.before_fusion_layer([cur_occ_feat] + prev_occ_feats)
        cur_occ_feat, prev_occ_feats = before_fuse_feats[0], before_fuse_feats[1:]
        ref_feat, prev_feat = self.t_fuse_mid(cur_occ_feat, prev_occ_feats)
        final_fused_feat = self.deform_fusion_layer(ref_feat, cur_occ_feat, prev_feat)
        output_feat = self.t_final(final_fused_feat)
        return output_feat
        # return mid_feat


class BeforeFusionLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.offset_conv = ConvModule(channels, channels,
            kernel_size=3, stride=1, padding=1,
            bias=False, conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d'))
    
    def forward(self, feat_list):
        return [self.offset_conv(feat) for feat in feat_list]


class TemporalDeformable(nn.Module):
    def __init__(self, channels, num_heads=4, num_samples=8):
        """
        channels: Number of input feature channels
        num_heads: Multi-head attention
        num_samples: Number of deformable sampling points per head
        """
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.num_samples = num_samples
        assert channels % num_heads == 0, "Channels must be divisible by num_heads"
        self.head_dim = channels // num_heads

        # Learnable offset generator (from F_t)
        self.offset_conv = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(channels, num_heads * num_samples * 3, kernel_size=3, padding=1, bias = False),
            nn.Tanh(),
        )

        # Key & Value projections for F_{t-1}
        # self.key_proj = nn.Conv3d(channels, channels, kernel_size=1)
        # self.value_proj = nn.Conv3d(channels, channels, kernel_size=1)
        self.key_value_proj = nn.Conv3d(channels, channels * 2, kernel_size=1)

        # Query projection for F_t
        self.query_proj = nn.Conv3d(channels, channels, kernel_size=1)

        # Output projection
        self.out_proj = nn.Conv3d(channels, channels, kernel_size=1)

        # Fusion gate
        # self.gate_conv = nn.Sequential(
        #     nn.Conv3d(2 * channels, channels, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(channels, channels, 3, padding=1),
        #     nn.Sigmoid()
        # )
        self.final_norm = nn.BatchNorm3d(channels)
        self.out_activate = nn.ReLU()

    def forward(self, feat_prev, feat_curr):
        B, C, D, H, W = feat_curr.shape
        device = feat_curr.device

        # Before fusion layer
        # feat_prev, feat_curr = self.before_fusion_layer(feat_prev, feat_curr)

        # Project key/value/query
        # key = self.key_proj(feat_prev)
        k_value = self.key_value_proj(feat_prev)
        query = self.query_proj(feat_curr)

        # Predict sampling offsets (B, H*N*3, D, H, W)
        offsets = self.offset_conv(feat_curr)
        offsets = offsets.view(B, self.num_heads, self.num_samples, 3, D, H, W)
        offsets = offsets.permute(0, 1, 4, 5, 6, 2, 3)  # (B, H, D, H, W, N, 3)

        # Create base grid
        z = torch.linspace(-1, 1, D, device=device)
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
        base_grid = torch.stack((zz, yy, xx), dim=-1)  # (D, H, W, 3)
        base_grid = base_grid[None, None].expand(B, self.num_heads, -1, -1, -1, -1)  # (B, H, D, H, W, 3)

        # Add offsets (normalized)
        sampling_grid = base_grid.unsqueeze(5) + offsets / torch.tensor(
            [D, H, W], device=device
        ).view(1, 1, 1, 1, 1, 1, 3)

        sampling_grid = sampling_grid.clamp(-1, 1)
        sampling_grid = sampling_grid.view(B * self.num_heads, D, H, W, self.num_samples, 3)
        sampling_grid = sampling_grid.permute(0, 4, 1, 2, 3, 5).reshape(
            B * self.num_heads * self.num_samples, D, H, W, 3
        )

        # Reshape value for multi-head
        k_value = k_value.view(B, self.num_heads, self.head_dim * 2, D, H, W)
        k_value = k_value.permute(0, 1, 3, 4, 5, 2).reshape(
            B * self.num_heads, 1, D, H, W, self.head_dim * 2)
        k_value_repeat = k_value.repeat(1, self.num_samples, 1, 1, 1, 1).reshape(
            B * self.num_heads * self.num_samples, D, H, W, self.head_dim * 2)

        # Sample value features
        # sampling_grid = sampling_grid.reshape(B * self.num_heads * self.num_samples, D, H, W, 3)
        sampled = F.grid_sample(
            k_value_repeat.permute(0, 4, 1, 2, 3),  # (B*H*N, C, D, H, W)
            sampling_grid,
            align_corners=True,
            mode='bilinear',
            padding_mode='border'
        )  # Output: (B*H*N, C, D, H, W)
        sampled = sampled.view(B, self.num_heads, self.num_samples, self.head_dim * 2, D, H, W)
        key, value = torch.chunk(sampled, chunks=2, dim=3)

        # Attention weights from query
        query = query.view(B, self.num_heads, self.head_dim, D, H, W)
        key = key.view(B, self.num_heads, self.num_samples, self.head_dim, D, H, W)
        query = query * (self.head_dim ** (-0.5))
        attn = torch.einsum("bmcdhw,bmscdhw->bmsdhw", query, key)  # (B, H, D, H, W, N)
        attn = F.softmax(attn, dim=2)

        # Weighted sum
        fused_feat = torch.einsum('bmsdhw,bmscdhw->bmcdhw', attn, value) # (B, H, C/H, D, H, W)
        fused_feat = fused_feat.contiguous().view(B, C, D, H, W)
        fused_feat = self.out_proj(fused_feat)

        out = self.out_activate(self.final_norm(fused_feat))

        # Gated fusion
        # fusion_input = torch.cat([feat_curr, fused_feat], dim=1)
        # gate = self.gate_conv(fusion_input)
        # out = gate * feat_curr + (1 - gate) * fused_feat
        return out


class AlignNetOcc3D(nn.Module):
    def __init__(self, clip_dim=1024, hsa_dim=240, embed_dim=384, clip_outdim=768,
                 layer_lifting_map=None, fusion_type="add", layer_depth=5, num_temporal=1):
        super(AlignNetOcc3D, self).__init__()
        x2side_map = {int(k): (int(i), int(j)) for i, j, k in [x.split("->") for x in layer_lifting_map]}
        self.fusion_map = x2side_map
        # Build fusion layers
        fusion_type = fusion_type
        self.fusion_layers = nn.ModuleDict(
            {
                f"layer_{tgt_idx}": build_fusion_layer_lift(
                    fusion_type, hsa_dim, clip_dim, embed_dim
                )
                for tgt_idx, (src_idx_clip, src_idx_san)  in x2side_map.items()
            }
        )

        self.layers_3d_body = nn.ModuleList([])
        for _ in range(layer_depth):
            self.layers_3d_body.append(
                ResBlock3D(channels_in=embed_dim, channels_out=embed_dim, use_checkpoint=False)
            )
        self.occupancy_pred = PredHead3DOcc(channels_in=embed_dim, channels_out=2, use_checkpoint=False)
        self.feat_pred = PredHead3DSem(channels_in=embed_dim, channels_out=clip_outdim, use_checkpoint=False)
        self.tf_layers = 0
        print(f"3D Align Network Layer Depth {layer_depth}, Temporal Fusion Layer {self.tf_layers}.")
        # self.temporal_fusion = TemporalFusionSimple(channels=embed_dim) if temporal else None
        self.temporal_fusion = TemporalFusionMultiFrame(
            channels=embed_dim, seqs=num_temporal - 1) if num_temporal > 1 else None
        # self.temporal_fusion = TemporalFusionMultiFrameCatOnly(channels=embed_dim) if temporal else None

    def forward(self, sem_feat: torch.tensor,
                clip_features: List, supp_features: List,
                depth: torch.Tensor, img_metas: List, 
                occ_feat_prevs: List[torch.Tensor] = None):

        # Prepare for 3d issues
        depth = self.prepare_depth(depth)
        if self.lss_view_transformer.mode == 'nuscenes':
            img_metas = self.prepare_meta(img_metas)
        h, w = clip_features[1].shape[2:]
        H, W = sem_feat.shape[2:]
        x = None
        if occ_feat_prevs is not None and len(occ_feat_prevs) == 0:
            occ_feat_prevs = None
        for idx, layer_3d in enumerate(self.layers_3d_body):
            x = self.fuse(idx, x, clip_features, supp_features, depth, img_metas, (h, w), (H, W))
            if idx == self.tf_layers and occ_feat_prevs is not None:
                # x = torch.cat([x, occ_feat_prev], dim=1)
                # x = checkpoint(self.temporal_fusion, x)
                x = checkpoint(self.temporal_fusion, x, occ_feat_prevs)
                # x = self.temporal_fusion(x, occ_feat_prevs)
            x = checkpoint(layer_3d, x)

        # Heads for occupy state and feat for clip
        bin_occ = checkpoint(self.occupancy_pred, x)
        feat_occ = checkpoint(self.feat_pred, x)
        out = {"bin_occ": bin_occ, "feat_occ": feat_occ}
        return out


    def forward_early(self, sem_feat: torch.tensor,
                clip_features: List, supp_features: List,
                depth: torch.Tensor, img_metas: List):

        # Prepare for 3d issues
        depth = self.prepare_depth(depth)
        if self.lss_view_transformer.mode == 'nuscenes':
            img_metas = self.prepare_meta(img_metas)
        h, w = clip_features[1].shape[2:]
        H, W = sem_feat.shape[2:]
        x = None
        x = self.fuse(0, x, clip_features, supp_features, depth, img_metas, (h, w), (H, W))
        return x

    def init_3d_sem(
            self,
            sem_feat: torch.tensor,
            depth: torch.Tensor,
            img_metas: List,
    ):
        sem_feat = self.sem_downsample_3d(sem_feat)
        feats_2d = self.prepare_feat_for_lifting(sem_feat)
        feats_lifted = self.lss_view_transformer([feats_2d] + img_metas, depth)
        return feats_lifted

    def fuse(
        self,
        block_idx: int,
        x: torch.Tensor,
        clip_features: List[torch.Tensor],
        supp_features: List[torch.Tensor],
        depth: torch.Tensor,
        img_metas: List,
        clip_shape: Tuple[int, int],
        lift_shape: Tuple[int, int]):

        if block_idx in self.fusion_map:
            src_idx_clip, src_idx_ec = self.fusion_map[block_idx]
            fused_feat = self.fusion_layers[f"layer_{block_idx}"](
                  supp_features[src_idx_ec], clip_features[src_idx_clip], lift_shape,
            )
            # fused_feat = self.fusion_layers[f"layer_{block_idx}"](
            #     clip_features[src_idx_clip], clip_shape,
            # )
            feats_2d = self.prepare_feat_for_lifting(fused_feat)
            feats_lifted = self.lss_view_transformer([feats_2d] + img_metas, depth)
            if x is not None:
                x = x + feats_lifted
            else:
                x = feats_lifted
        return x

    def prepare_depth(self, depth: torch.Tensor):
        depth_ds = self.lss_view_transformer.downsample_depth(depth, downsample=8)
        # TODO: one_hot depth is used for LSS visualization test, two_hot depth for practical LSS
        # one_hot_depth = self.lss_view_transformer.get_one_hot_depth(depth_ds)
        # return one_hot_depth
        two_hot_depth = self.lss_view_transformer.get_two_hot_depth(depth_ds)
        return two_hot_depth

    def prepare_meta(self, img_metas):

        N = self.num_camera
        sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = img_metas
        sensor2egos = sensor2egos.view(-1, self.num_frame, N, 4, 4)
        ego2globals = ego2globals.view(-1, self.num_frame, N, 4, 4)

        # Calculate the transformation from sweep sensor to key ego
        keyego2global = ego2globals[:, 0, 0, ...].unsqueeze(1).unsqueeze(1)
        global2keyego = torch.inverse(keyego2global.double())
        sensor2keyegos = \
            global2keyego @ ego2globals.double() @ sensor2egos.double()
        sensor2keyegos = sensor2keyegos.float()

        extra = [
            sensor2keyegos,
            ego2globals,
            intrins.view(-1, self.num_frame, N, 3, 3),
            post_rots.view(-1, self.num_frame, N, 3, 3),
            post_trans.view(-1, self.num_frame, N, 3)
        ]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        sensor2keyegos, ego2globals, intrins, post_rots, post_trans = extra
        return [sensor2keyegos[0], ego2globals[0], intrins[0], post_rots[0], post_trans[0], bda[0]]

    def prepare_feat_for_lifting(self, feats_2d):
        _, C, H, W = feats_2d.shape
        N = self.num_camera
        feats_2d = feats_2d.view(-1, N, self.num_frame, C, H, W)
        feats_2d = torch.split(feats_2d, 1, dim=2)
        feats_2d = [t.squeeze(2) for t in feats_2d]
        return feats_2d[0]


class ResBlock3D(nn.Module):
    def __init__(self, channels_in, channels_out, stride=1, downsample=None, use_checkpoint=False):
        super(ResBlock3D, self).__init__()
        self.conv1 = ConvModule(
            channels_in,
            channels_out,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d', ),
            act_cfg=dict(type='ReLU', inplace=True))
        self.conv2 = ConvModule(
            channels_out,
            channels_out,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d', ),
            act_cfg=None)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x
        if self.use_checkpoint:
            x = checkpoint(self.conv1, x)
        else:
            x = self.conv1(x)
        if self.use_checkpoint:
            x = checkpoint(self.conv2, x)
        else:
            x = self.conv2(x)
        x = x + identity
        return self.relu(x)


class PredHead3D(nn.Module):
    def __init__(self, channels_in, channels_out, channels_mid=None, stride=1, use_checkpoint=False):
        super(PredHead3D, self).__init__()
        self.occ_pred = nn.ModuleList([])
        channels = [channels_in] + channels_mid + [channels_out]
        for i in range(1, len(channels_mid)):
            layer = ConvModule(
                channels[i - 1],
                channels[i],
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
                conv_cfg=dict(type='Conv3d'))
                # norm_cfg=dict(type='BN3d', ),
                # act_cfg=dict(type='ReLU',inplace=True))
            self.occ_pred.append(layer)

        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        if self.use_checkpoint:
            return checkpoint(self.occ_pred, x)
        else:
            return self.occ_pred(x)

class PredHead3DOcc(nn.Module):
    def __init__(self, channels_in, channels_out, stride=1, use_checkpoint=False):
        super(PredHead3DOcc, self).__init__()
        channels_mid = channels_in // 4
        self.occ_conv1 = ConvModule(
                channels_in,
                channels_mid,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
                conv_cfg=dict(type='Conv3d'),
                norm_cfg=dict(type='BN3d'),
                # act_cfg=None,
                act_cfg=dict(type='ReLU', inplace=True),
        )

        self.occ_conv2 = ConvModule(
                channels_mid,
                channels_out,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
                conv_cfg=dict(type='Conv3d'),
                act_cfg=None,
                # act_cfg=dict(type='ReLU'),
        )

        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        if self.use_checkpoint:
            x = checkpoint(self.occ_conv1, x)
        else:
            x = self.occ_conv1(x)
        if self.use_checkpoint:
            x = checkpoint(self.occ_conv2, x)
        else:
            x = self.occ_conv2(x)
        return x


class PredHead3DSem(nn.Module):
    def __init__(self, channels_in, channels_out, stride=1, use_checkpoint=False):
        super(PredHead3DSem, self).__init__()
        channels_mid = channels_in # (channels_in + channels_out) // 2
        self.occ_conv1 = ConvModule(
            channels_in,
            channels_in,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=True,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d'),
            # act_cfg=None,
            act_cfg=dict(type='ReLU', inplace=True),
        )

        self.occ_conv2 = ConvModule(
            channels_in,
            channels_mid,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=False,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d', ),
            # act_cfg=None,
            act_cfg=dict(type='ReLU', inplace=True),
        )

        self.occ_conv3 = ConvModule(
            channels_mid,
            channels_out,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=False,
            conv_cfg=dict(type='Conv3d'),
            act_cfg=None,
        )

        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        if self.use_checkpoint:
            x = checkpoint(self.occ_conv1, x)
        else:
            x = self.occ_conv1(x)
        if self.use_checkpoint:
            x = checkpoint(self.occ_conv2, x)
        else:
            x = self.occ_conv2(x)
        if self.use_checkpoint:
            x = checkpoint(self.occ_conv3, x)
        else:
            x = self.occ_conv3(x)
        # Make sure x in (-0.5, 0.5)
        x = x.sigmoid() - 0.5
        return x