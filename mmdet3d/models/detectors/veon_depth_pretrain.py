# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import random

from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from mmdet.models import DETECTORS
from .base import Base3DDetector
from .. import builder
from mmdet3d.apis.train import count_parameters, count_parameters_depth
from mmcv.cnn.bricks.conv_module import ConvModule
from mmdet3d.utils.vis import vis_img_depth, vis_img_normal, vis_occ


@DETECTORS.register_module()
class VeonDepthPretrain(Base3DDetector):

    def __init__(
        self,
        image_encoder = None,
        depth_estimator = None,
        img_view_transformer = None,
        prompt_encoder = None,
        mask_decoder = None,
        loss_occ = None,
        use_mask = True,
        num_classes = 18,
        num_adj = 0,
        align_after_view_transfromation=False,
        pretrained=None,
        with_prev=True,
        train_cfg = None,
        test_cfg = None,
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
        super(VeonDepthPretrain, self).__init__(**kwargs)
        self.depth_estimator = builder.build_neck(depth_estimator)
        self.img_view_transformer = builder.build_neck(img_view_transformer)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.num_classes = num_classes
        self.use_mask = use_mask
        self.num_frame = 1
        self.with_prev = with_prev
        self.depth_mode = depth_mode
        self.nonce = 0
        self.avg_depth_error = 0.0
        self.pred_depth_scale = 8
        self.gt_depth_scale = 16
        self._freeze_stages()


    def train(self, mode=True):
        """
        Convert the model into training mode
        """
        super(VeonDepthPretrain, self).train(mode)
        self._freeze_stages()
        count_parameters_depth(self)


    def _freeze_stages(self):

        for name, param in self.depth_estimator.named_parameters():
            if 'pretrain' in name and 'lora' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True


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

        h, w = img_inputs[0].shape[-2:]
        depth_out = self.estimate_depth(
            depth_input=kwargs['depth_img_inputs'], 
            depth_size=(h // 2, w // 2), 
        )
        depth = depth_out['metric_depth']

        # Downsample predicted depth and gt depth
        depth_ds = self.img_view_transformer.downsample_depth(depth, downsample=self.pred_depth_scale)
        gt_depth_ds = self.img_view_transformer.downsample_depth(kwargs['gt_depth'], downsample=self.gt_depth_scale)

        # Do statistics for average depth error
        depth_flatten, gt_depth_flatten = depth_ds.view(-1), gt_depth_ds.view(-1)
        valid_tag = gt_depth_flatten < 9225
        depth_flatten, gt_depth_flatten = depth_flatten[valid_tag], gt_depth_flatten[valid_tag]
        depth_error = torch.abs(depth_flatten - gt_depth_flatten).mean().item()
        self.avg_depth_error = (self.nonce * self.avg_depth_error + depth_error) / (self.nonce + 1)
        self.nonce += 1
        if random.randint(0, 200) >= 200:
            print(f'current depth error：{depth_error}, average depth error: {self.avg_depth_error}.')

        # Loss
        loss_depth = self.img_view_transformer.get_depth_loss_own(gt_depth_ds, depth_ds, zoe=True, ce=True)
        losses = dict()
        losses.update(loss_depth)
        # torch.cuda.empty_cache()
        return losses


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


    def extract_img_feat(self,
                         img,
                         img_metas,
                         depth=None,
                         pred_prev=False,
                         sequential=False,
                         **kwargs):

        raise NotImplementedError

    def extract_feat(self, feat_inputs, **kwargs):
        raise NotImplementedError

    def loss_single(self,voxel_semantics, mask_camera, preds):
        raise NotImplementedError

    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img_inputs=None,
                     **kwargs):
        raise NotImplementedError

    def aug_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        raise NotImplementedError

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        raise NotImplementedError
