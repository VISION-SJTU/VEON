# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import random, os

from torch import nn
import torch
from torch.nn import functional as F
import numpy as np
from sklearn.metrics import average_precision_score
from mmdet.models import DETECTORS
from .base import Base3DDetector
from .. import builder
from mmdet3d.apis.train import count_parameters, count_parameters_full
from mmcv.cnn.bricks.conv_module import ConvModule
from mmdet3d.utils.vis import vis_img_depth, vis_img_normal, vis_occ, visualize_camera_images


@DETECTORS.register_module()
class VeonDepthCache(Base3DDetector):

    def __init__(
        self,
        semantic_model = None,
        depth_estimator=None,
        img_view_transformer = None,
        loss_occ = None,
        use_mask = True,
        num_classes = 18,
        num_adj = 0,
        align_after_view_transfromation=False,
        pretrained=None,
        with_prev=True,
        train_cfg = None,
        test_cfg = None,
        mode="nuscenes",
        depth_pred_home=None,
        depth_mode = "zoedepth",
        **kwargs
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super(VeonDepthCache, self).__init__(**kwargs)
        self.depth_estimator = builder.build_neck(depth_estimator)
        self.img_view_transformer = builder.build_neck(img_view_transformer)
        self.fake_weight = nn.Linear(1, 1)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.num_classes = num_classes
        self.use_mask = use_mask
        self.mode = mode
        self.loss_occ = builder.build_loss(loss_occ)

        self.num_frame = 1
        self.with_prev = with_prev
        self.depth_mode = depth_mode
        self._freeze_stages()
        self.nonce = 0
        self.depth_pred_home = depth_pred_home
        self.avg_depth_error = 0.0

    def train(self, mode=True):
        """
        Convert the model into training mode
        """
        super(VeonDepthCache, self).train(mode)
        self._freeze_stages()
        # count_parameters(self)
        count_parameters_full(self)

    def _freeze_stages(self):
        for name, param in self.depth_estimator.named_parameters():
            param.requires_grad = False

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """

        with torch.no_grad():

            h, w = img_inputs[0].shape[-2:]
            depth_out = self.estimate_depth(
                depth_input=kwargs['depth_img_inputs'],
                depth_size=(h // 2, w // 2),
            )
            depth = depth_out['metric_depth']

            # Codes for avg depth error
            depth_ds = self.img_view_transformer.downsample_depth(depth, downsample=8)
            gt_depth_ds = self.img_view_transformer.downsample_depth(kwargs['gt_depth'], downsample=16)
            depth_flatten, gt_depth_flatten = depth_ds.view(-1), gt_depth_ds.view(-1)
            valid_tag = gt_depth_flatten < 9225
            depth_flatten, gt_depth_flatten = depth_flatten[valid_tag], gt_depth_flatten[valid_tag]
            depth_error = torch.abs(depth_flatten - gt_depth_flatten).mean().item()
            self.avg_depth_error = (self.nonce * self.avg_depth_error + depth_error) / (self.nonce + 1)
            if random.randint(0, 200) >= 200:
                print(f'current depth error：{depth_error}, average depth error: {self.avg_depth_error}.')

            # For depth offline storing
            # Typically, we use "./data/nuscenes/depth_cache/depth" for base folder of depth cache
            base_folder = self.depth_pred_home 
            for i, token in enumerate(img_metas[0]['unique_tokens']):
                tk, cam_name = token.split("-")
                os.makedirs(os.path.join(base_folder, tk[:2], tk), exist_ok=True)
                depth_name = os.path.join(base_folder, tk[:2], tk, token + ".tensor")
                if os.path.exists(depth_name):
                    continue
                if random.randint(0, 200) >= 200:
                    print("Saving {}".format(token))
                torch.save(depth[0][i].cpu(), depth_name)

                # depth_recover = torch.load(depth_name)
                # print("error: ", torch.abs(depth[0][i].cpu() - depth_recover).sum())

        losses = dict()
        fake_out = self.fake_weight(depth.mean().unsqueeze(0).detach())
        losses["fake_loss"] = 0.0 * torch.abs(fake_out - 1)
        self.nonce += 1

        return losses

    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img_inputs=None,
                     **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        for var, name in [(img_inputs, 'img_inputs'),
                          (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(img_inputs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(img_inputs), len(img_metas)))

        if not isinstance(img_inputs[0][0], list):
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            return self.simple_test(points[0], img_metas[0], img_inputs[0],
                                    **kwargs)
        else:
            return self.aug_test(None, img_metas[0], img_inputs[0], **kwargs)

    def aug_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentation."""
        assert False

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentation."""
        raise NotImplementedError

    def estimate_depth(self, depth_input, depth_size):
        B, N, C, H, W = depth_input.shape
        din = depth_input.view(-1, C, H, W)
        dout = self.depth_estimator(din)
        abs_depth = dout['metric_depth']
        if (abs_depth.shape[-2], abs_depth.shape[-1]) != depth_size:
            abs_depth = F.interpolate(abs_depth[:, None], depth_size, mode="bilinear", align_corners=True)
        _, Co, Ho, Wo = abs_depth.shape
        dout['metric_depth'] = abs_depth.view(B, N, Co, Ho, Wo).squeeze(dim=2)
        return dout

    def pass_semantic_model(self, img_inputs, depth):
        imgs = img_inputs[0]
        img_metas = img_inputs[1:]
        output_dict = self.semantic_model(imgs, depth, img_metas)
        return output_dict

    def extract_feat(self, feat_inputs, depth=None, **kwargs):
        raise NotImplementedError
