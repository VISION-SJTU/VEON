# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer
from mmcv.runner import BaseModule, force_fp32
from torch.cuda.amp.autocast_mode import autocast
from torch.utils.checkpoint import checkpoint

from mmdet3d.ops.bev_pool_v2.bev_pool import bev_pool_v2
from mmdet.models.backbones.resnet import BasicBlock
from ..builder import NECKS
from einops import rearrange
import math


@NECKS.register_module()
class LSSViewTransformerRaw(BaseModule):
    r"""Lift-Splat-Shoot view transformer with BEVPoolv2 implementation.

    Please refer to the `paper <https://arxiv.org/abs/2008.05711>`_ and
        `paper <https://arxiv.org/abs/2211.17111>`

    Args:
        grid_config (dict): Config of grid alone each axis in format of
            (lower_bound, upper_bound, interval). axis in {x,y,z,depth}.
        input_size (tuple(int)): Size of input images in format of (height,
            width).
        downsample (int): Down sample factor from the input size to the feature
            size.
        in_channels (int): Channels of input feature.
        out_channels (int): Channels of transformed feature.
        accelerate (bool): Whether the view transformation is conducted with
            acceleration. Note: the intrinsic and extrinsic of cameras should
            be constant when 'accelerate' is set true.
        sid (bool): Whether to use Spacing Increasing Discretization (SID)
            depth distribution as `STS: Surround-view Temporal Stereo for
            Multi-view 3D Detection`.
        collapse_z (bool): Whether to collapse in z direction.
    """

    def __init__(
        self,
        grid_config,
        input_size,
        downsample=16,
        out_channels=256,
        accelerate=False,
        sid=False,
        collapse_z=True,
        mode="nuscenes",
        loss_depth_weight=0.05,
        ds_feat=[2, 2, 2],
    ):
        super(LSSViewTransformerRaw, self).__init__()
        self.grid_config = grid_config
        self.downsample = downsample
        self.create_grid_infos(**grid_config)
        self.sid = sid
        self.frustum = self.create_frustum(grid_config['depth'],
                                           input_size, downsample)
        self.out_channels = out_channels
        self.accelerate = accelerate
        self.initial_flag = True
        self.collapse_z = collapse_z
        self.loss_depth_weight = loss_depth_weight
        self.mode = mode
        self.ds = ds_feat
        assert len(self.ds) == 3
        self.use_ds = any(x != 1 for x in self.ds)
        assert self.mode in ['nuscenes']
        # self.to_dist = nn.Conv2d(90, 90, kernel_size=1, stride=1, bias=True)

    def create_grid_infos(self, x, y, z, **kwargs):
        """Generate the grid information including the lower bound, interval,
        and size.

        Args:
            x (tuple(float)): Config of grid alone x axis in format of
                (lower_bound, upper_bound, interval).
            y (tuple(float)): Config of grid alone y axis in format of
                (lower_bound, upper_bound, interval).
            z (tuple(float)): Config of grid alone z axis in format of
                (lower_bound, upper_bound, interval).
            **kwargs: Container for other potential parameters
        """
        self.grid_lower_bound = torch.Tensor([cfg[0] for cfg in [x, y, z]])
        self.grid_interval = torch.Tensor([cfg[2] for cfg in [x, y, z]])
        self.grid_size = torch.Tensor([(cfg[1] - cfg[0]) / cfg[2] for cfg in [x, y, z]])

    def create_frustum(self, depth_cfg, input_size, downsample):
        """Generate the frustum template for each image.

        Args:
            depth_cfg (tuple(float)): Config of grid alone depth axis in format
                of (lower_bound, upper_bound, interval).
            input_size (tuple(int)): Size of input images in format of (height,
                width).
            downsample (int): Down sample scale factor from the input size to
                the feature size.
        """
        H_in, W_in = input_size
        H_feat, W_feat = H_in // downsample, W_in // downsample
        d = torch.arange(*depth_cfg, dtype=torch.float)\
            .view(-1, 1, 1).expand(-1, H_feat, W_feat)
        self.D = d.shape[0]
        if self.sid:
            d_sid = torch.arange(self.D).float()
            depth_cfg_t = torch.tensor(depth_cfg).float()
            d_sid = torch.exp(torch.log(depth_cfg_t[0]) + d_sid / (self.D-1) *
                              torch.log((depth_cfg_t[1]-1) / depth_cfg_t[0]))
            d = d_sid.view(-1, 1, 1).expand(-1, H_feat, W_feat)
        x = torch.linspace(0, W_in - 1, W_feat,  dtype=torch.float)\
            .view(1, 1, W_feat).expand(self.D, H_feat, W_feat)
        y = torch.linspace(0, H_in - 1, H_feat,  dtype=torch.float)\
            .view(1, H_feat, 1).expand(self.D, H_feat, W_feat)

        # D x H x W x 3
        return torch.stack((x, y, d), -1)

    def get_lidar_coor(self, sensor2ego, ego2global, cam2imgs, post_rots, post_trans, bda):
        """Calculate the locations of the frustum points in the lidar
        coordinate system.

        Args:
            rots (torch.Tensor): Rotation from camera coordinate system to
                lidar coordinate system in shape (B, N_cams, 3, 3).
            trans (torch.Tensor): Translation from camera coordinate system to
                lidar coordinate system in shape (B, N_cams, 3).
            cam2imgs (torch.Tensor): Camera intrinsic matrixes in shape
                (B, N_cams, 3, 3).
            post_rots (torch.Tensor): Rotation in camera coordinate system in
                shape (B, N_cams, 3, 3). It is derived from the image view
                augmentation.
            post_trans (torch.Tensor): Translation in camera coordinate system
                derived from image view augmentation in shape (B, N_cams, 3).

        Returns:
            torch.tensor: Point coordinates in shape
                (B, N_cams, D, ownsample, 3)
        """
        B, N, _, _ = sensor2ego.shape

        # post-transformation
        # B x N x D x H x W x 3
        points = self.frustum.to(sensor2ego) - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3) \
            .matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat(
            (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)
        combine = sensor2ego[:,:,:3,:3].matmul(torch.inverse(cam2imgs))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += sensor2ego[:,:,:3, 3].view(B, N, 1, 1, 1, 3)
        points = bda.view(B, 1, 1, 1, 1, 3,
                          3).matmul(points.unsqueeze(-1)).squeeze(-1)
        return points

    def get_lidar_coor_semkitti(self, rots, trans, intrins, post_rots, post_trans, bda):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape
        self.frustum = self.frustum.to(post_trans)

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)

        if intrins.shape[3] == 4:  # for KITTI
            shift = intrins[:, :, :3, 3]
            points = points - shift.view(B, N, 1, 1, 1, 3, 1)
            intrins = intrins[:, :, :3, :3]

        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        if bda.shape[-1] == 4:
            points = torch.cat((points, torch.ones(*points.shape[:-1], 1).type_as(points)), dim=-1)
            points = bda.view(B, 1, 1, 1, 1, 4, 4).matmul(points.unsqueeze(-1)).squeeze(-1)
            points = points[..., :3]
        else:
            points = bda.view(B, 1, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)

        return points

    def init_acceleration_v2(self, coor):
        """Pre-compute the necessary information in acceleration including the
        index of points in the final feature.

        Args:
            coor (torch.tensor): Coordinate of points in lidar space in shape
                (B, N_cams, D, H, W, 3).
            x (torch.tensor): Feature of points in shape
                (B, N_cams, D, H, W, C).
        """

        ranks_bev, ranks_depth, ranks_feat, \
            interval_starts, interval_lengths = \
            self.voxel_pooling_prepare_v2(coor)

        self.ranks_bev = ranks_bev.int().contiguous()
        self.ranks_feat = ranks_feat.int().contiguous()
        self.ranks_depth = ranks_depth.int().contiguous()
        self.interval_starts = interval_starts.int().contiguous()
        self.interval_lengths = interval_lengths.int().contiguous()

    def voxel_pooling_v2(self, coor, depth, feat):
        ranks_bev, ranks_depth, ranks_feat, \
            interval_starts, interval_lengths = \
            self.voxel_pooling_prepare_v2(coor)
        if ranks_feat is None:
            print('warning ---> no points within the predefined '
                  'bev receptive field')
            dummy = torch.zeros(size=[
                feat.shape[0], feat.shape[2],
                int(self.grid_size[2]),
                int(self.grid_size[0]),
                int(self.grid_size[1])
            ]).to(feat)
            dummy = torch.cat(dummy.unbind(dim=2), 1)
            return dummy
        feat = feat.permute(0, 1, 3, 4, 2)
        bev_feat_shape = (depth.shape[0], int(self.grid_size[2]),
                          int(self.grid_size[1]), int(self.grid_size[0]),
                          feat.shape[-1])  # (B, Z, Y, X, C)
        bev_feat = bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                               bev_feat_shape, interval_starts,
                               interval_lengths)
        # collapse Z
        if self.collapse_z:
            bev_feat = torch.cat(bev_feat.unbind(dim=2), 1)
        return bev_feat

    def voxel_pooling_prepare_v2(self, coor):
        """Data preparation for voxel pooling.

        Args:
            coor (torch.tensor): Coordinate of points in the lidar space in
                shape (B, N, D, H, W, 3).

        Returns:
            tuple[torch.tensor]: Rank of the voxel that a point is belong to
                in shape (N_Points); Reserved index of points in the depth
                space in shape (N_Points). Reserved index of points in the
                feature space in shape (N_Points).
        """
        B, N, D, H, W, _ = coor.shape
        num_points = B * N * D * H * W
        # record the index of selected points for acceleration purpose
        ranks_depth = torch.range(
            0, num_points - 1, dtype=torch.int, device=coor.device)
        ranks_feat = torch.range(
            0, num_points // D - 1, dtype=torch.int, device=coor.device)
        ranks_feat = ranks_feat.reshape(B, N, 1, H, W)
        ranks_feat = ranks_feat.expand(B, N, D, H, W).flatten()
        # convert coordinate into the voxel space
        coor = ((coor - self.grid_lower_bound.to(coor)) /
                self.grid_interval.to(coor))
        coor = coor.long().view(num_points, 3)
        batch_idx = torch.range(0, B - 1).reshape(B, 1). \
            expand(B, num_points // B).reshape(num_points, 1).to(coor)
        coor = torch.cat((coor, batch_idx), 1)

        # filter out points that are outside box
        kept = (coor[:, 0] >= 0) & (coor[:, 0] < self.grid_size[0]) & \
               (coor[:, 1] >= 0) & (coor[:, 1] < self.grid_size[1]) & \
               (coor[:, 2] >= 0) & (coor[:, 2] < self.grid_size[2])
        if len(kept) == 0:
            return None, None, None, None, None
        coor, ranks_depth, ranks_feat = \
            coor[kept], ranks_depth[kept], ranks_feat[kept]
        # get tensors from the same voxel next to each other
        ranks_bev = coor[:, 3] * (
            self.grid_size[2] * self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 2] * (self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 1] * self.grid_size[0] + coor[:, 0]
        order = ranks_bev.argsort()
        ranks_bev, ranks_depth, ranks_feat = \
            ranks_bev[order], ranks_depth[order], ranks_feat[order]

        kept = torch.ones(
            ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
        interval_starts = torch.where(kept)[0].int()
        if len(interval_starts) == 0:
            return None, None, None, None, None
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
        return ranks_bev.int().contiguous(), ranks_depth.int().contiguous(
        ), ranks_feat.int().contiguous(), interval_starts.int().contiguous(
        ), interval_lengths.int().contiguous()

    def pre_compute(self, input):
        if self.initial_flag:
            coor = self.get_lidar_coor(*input[1:7])
            self.init_acceleration_v2(coor)
            self.initial_flag = False

    def view_transform_core(self, input, depth, tran_feat):
        B, N, C, H, W = input[0].shape

        # Lift-Splat
        if self.accelerate:
            feat = tran_feat.view(B, N, self.out_channels, H, W)
            feat = feat.permute(0, 1, 3, 4, 2)
            depth = depth.view(B, N, self.D, H, W)
            bev_feat_shape = (depth.shape[0], int(self.grid_size[2]),
                              int(self.grid_size[1]), int(self.grid_size[0]),
                              feat.shape[-1])  # (B, Z, Y, X, C)
            bev_feat = bev_pool_v2(depth, feat, self.ranks_depth,
                                   self.ranks_feat, self.ranks_bev,
                                   bev_feat_shape, self.interval_starts,
                                   self.interval_lengths)

            bev_feat = bev_feat.squeeze(2)
        else:
            coor = self.get_lidar_coor(*input[1:7])
            bev_feat = self.voxel_pooling_v2(
                coor, depth.view(B, N, self.D, H, W),
                tran_feat.view(B, N, self.out_channels, H, W))
        return bev_feat

    def view_transform(self, input, depth, tran_feat):
        if self.accelerate:
            self.pre_compute(input)
        return self.view_transform_core(input, depth, tran_feat)

    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N, H // self.downsample,
                                   self.downsample, W // self.downsample,
                                   self.downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample,
                                   W // self.downsample)

        if not self.sid:
            gt_depths = (gt_depths - (self.grid_config['depth'][0] -
                                      self.grid_config['depth'][2])) / \
                        self.grid_config['depth'][2]
        else:
            gt_depths = torch.log(gt_depths) - torch.log(
                torch.tensor(self.grid_config['depth'][0]).float())
            gt_depths = gt_depths * (self.D - 1) / torch.log(
                torch.tensor(self.grid_config['depth'][1] - 1.).float() /
                self.grid_config['depth'][0])
            gt_depths = gt_depths + 1.
        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0),
                                gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(
            gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:, 1:]
        return gt_depths.float()

    def get_absolute_depth(self, depths):
        bin_centers = torch.arange(self.D + 2, device=depths.device) * self.grid_config['depth'][2] + \
                      (self.grid_config['depth'][0] - self.grid_config['depth'][2] / 2)
        abs_depth_avg = depths * bin_centers[None, None, None, :]
        abs_depth_avg = abs_depth_avg.sum(dim=-1)

        def hard_max(logits, dim):
            # y_soft = torch.softmax(logits, dim=dim)
            index = logits.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
            # ret = y_hard - y_soft.detach() + y_soft
            return y_hard
        depths_hard = hard_max(depths, dim=-1)
        abs_depth_hard = depths_hard * bin_centers[None, None, None, :]
        abs_depth_hard = abs_depth_hard.sum(dim=-1)
        return abs_depth_avg, abs_depth_hard

    def downsample_depth(self, depths, downsample):
        B, N, H, W = depths.shape
        depths = depths.view(B * N, H // downsample,
                             downsample, W // downsample,
                             downsample, 1)
        depths = depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        depths = depths.view(-1, downsample * downsample)
        depths_tmp = torch.where(depths == 0.0,
                                 1e5 * torch.ones_like(depths), depths)
        depths = torch.min(depths_tmp, dim=-1).values
        depths = depths.view(B, N, H // downsample, W // downsample)
        return depths

    def get_two_hot_depth(self, depths, gamma=4, downsample=False):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        if downsample:
            depths = self.downsample_depth(depths, self.downsample)
        B, N, H, W = depths.shape
        depths = depths.view(B * N, H, W)
        bin_centers = torch.arange(self.D + 1, device=depths.device) * self.grid_config['depth'][2] + \
                      (self.grid_config['depth'][0] + self.grid_config['depth'][2] / 2)
        depths = depths.view(-1, H, W, 1).repeat(1, 1, 1, self.D + 1)
        gap = -torch.abs(depths - bin_centers[None, None, None, :]) * gamma
        MIN_GAP = -16
        gap = torch.where(gap >= MIN_GAP, gap, gap + (MIN_GAP - gap.detach()))
        # gap = torch.permute(gap, (0, 3, 1, 2))
        # depth_dist = self.to_dist(gap).permute((0, 2, 3, 1))
        depth_dist = torch.softmax(gap, dim=-1)
        # depth_dist = torch.clamp_min(depth_dist, min=1e-7)
        depth_dist = depth_dist.view(-1, self.D + 1)[:, :-1]
        depth_dist = depth_dist.view(B, N, H, W, self.D).permute(0, 1, 4, 2, 3)
        return depth_dist

    def get_one_hot_depth(self, depths, downsample=False):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        if downsample:
            depths = self.downsample_depth(depths, self.downsample)
        B, N, H, W = depths.shape
        depths = depths.view(B * N, H, W).clamp_max(max=500)
        bin_centers = torch.arange(self.D + 1, device=depths.device) * self.grid_config['depth'][2] + \
                      (self.grid_config['depth'][0] + self.grid_config['depth'][2] / 2)
        depths = depths.view(-1, H, W, 1).repeat(1, 1, 1, self.D + 1)
        gap = -torch.abs(depths - bin_centers[None, None, None, :])
        # depth_dist = torch.nn.functional.gumbel_softmax(gap, tau=1.0, hard=True)
        def hard_max(logits, dim):
            # y_soft = torch.softmax(logits, dim=dim)
            index = logits.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
            # ret = y_hard - y_soft.detach() + y_soft
            return y_hard
        depth_dist = hard_max(gap, dim=-1)
        depth_dist = depth_dist.view(-1, self.D + 1)[:, :-1]
        depth_dist = depth_dist.view(B, N, H, W, self.D).permute(0, 1, 4, 2, 3)
        return depth_dist

    def get_one_hot_depth_gumbel(self, depths, downsample=False, gamma=5):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        if downsample:
            depths = self.downsample_depth(depths, self.downsample)
        B, N, H, W = depths.shape
        depths = depths.view(B * N, H, W).clamp_max(max=500)
        bin_centers = torch.arange(self.D + 1, device=depths.device) * self.grid_config['depth'][2] + \
                      (self.grid_config['depth'][0] + self.grid_config['depth'][2] / 2)
        depths = depths.view(-1, H, W, 1).repeat(1, 1, 1, self.D + 1)
        gap = -torch.abs(depths - bin_centers[None, None, None, :]) * gamma
        gap_prob = torch.softmax(gap, dim=-1)
        depth_dist = torch.nn.functional.gumbel_softmax(gap_prob, tau=1.0, hard=True)
        depth_dist = depth_dist.view(-1, self.D + 1)[:, :-1]
        depth_dist = depth_dist.view(B, N, H, W, self.D).permute(0, 1, 4, 2, 3)
        return depth_dist

    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds):
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        if len(depth_preds.shape) == 5:
            B, N, C, H, W = depth_preds.shape
            depth_preds = depth_preds.reshape(-1, C, H, W)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(
                depth_preds,
                depth_labels,
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum())
        return self.loss_depth_weight * depth_loss

    @force_fp32()
    def get_depth_loss_own(self, depth_labels_orig, depth_preds_orig, zoe=True, ce=True):

        loss = dict()

        if zoe:
            depth_flatten, gt_depth_flatten = depth_preds_orig.view(-1), depth_labels_orig.view(-1)
            valid_tag = gt_depth_flatten < 9225
            depth_flatten, gt_depth_flatten = depth_flatten[valid_tag], gt_depth_flatten[valid_tag]
            with autocast(enabled=False):
                alpha = 1e-7
                g = torch.log(depth_flatten + alpha) - torch.log(gt_depth_flatten + alpha)
                Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
                loss_zoe = torch.clip(torch.sqrt(Dg), max=2.0)
                loss['loss_depth_zoe'] = loss_zoe

        if ce:
            depth_labels = self.get_one_hot_depth(depth_labels_orig)
            depth_preds = self.get_two_hot_depth(depth_preds_orig)
            if len(depth_preds.shape) == 5:
                B, N, C, H, W = depth_preds.shape
                depth_preds = depth_preds.reshape(-1, C, H, W)
            if len(depth_labels.shape) == 5:
                B, N, C, H, W = depth_labels.shape
                depth_labels = depth_labels.reshape(-1, C, H, W)
            depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D)
            depth_labels = depth_labels.permute(0, 2, 3, 1).contiguous().view(-1, self.D)
            fg_mask = torch.max(depth_labels, dim=1).values > 0.0
            depth_labels = depth_labels[fg_mask]
            depth_preds = depth_preds[fg_mask]
            with autocast(enabled=False):
                depth_loss_ce = F.binary_cross_entropy(
                    depth_preds,
                    depth_labels,
                    reduction='none',
                ).sum() / max(1.0, fg_mask.sum())
                loss['loss_depth_ce'] = depth_loss_ce * 0.05

        return loss

    def forward(self, input, depth, stereo_metas=None):
        # NuScenes
        (tran_feat, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda) = input[:7]

        B, N, C, H, W = tran_feat.shape
        tran_feat = tran_feat.view(B * N, C, H, W)
        B, N, C, H, W = depth.shape
        depth = depth.view(B * N, C, H, W)

        # TODO: make sure that depth is in logit format
        bev_feat = self.view_transform(input, depth, tran_feat)

        if self.use_ds:
            bev_feat = rearrange(bev_feat,
                                 'b c (z dz) (h dh) (w dw) -> b c z h w (dz dh dw)',
                                 dz=self.ds[0], dh=self.ds[1], dw=self.ds[2])
            bev_feat = torch.max(bev_feat, dim=-1).values

        return bev_feat

