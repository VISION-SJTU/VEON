# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import random

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
from torch.utils.checkpoint import checkpoint

@DETECTORS.register_module()
class VeonTemporal(Base3DDetector):

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
        retrieval=False,
        use_depth_estimator=False,
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
        super(VeonTemporal, self).__init__(**kwargs)
        self.depth_estimator = builder.build_neck(depth_estimator) if \
              use_depth_estimator or (test_cfg is not None and test_cfg['depth_estimator']) else None
        self.img_view_transformer = builder.build_neck(img_view_transformer)
        self.semantic_model = builder.build_neck(semantic_model)
        self.semantic_model.prepare_lss(self.img_view_transformer)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.num_classes = num_classes
        self.use_mask = use_mask
        self.mode = mode
        self.loss_occ = builder.build_loss(loss_occ)

        self.num_frame = 1
        self.num_cam = 6
        self.with_prev = with_prev
        self.depth_mode = depth_mode
        self._freeze_stages()
        self.nonce = 1
        self.retrieval = retrieval

    def train(self, mode=True):
        """
        Convert the model into training mode
        """
        super(VeonTemporal, self).train(mode)
        self._freeze_stages()
        # count_parameters(self)
        count_parameters_full(self)

    def _freeze_stages(self):

        for name, param in self.semantic_model.model.ov_classifier.named_parameters():
            param.requires_grad = False
        for name, param in self.semantic_model.model.side_adapter_network.named_parameters():
            param.requires_grad = False
        if self.depth_estimator is not None:
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

        h, w = img_inputs[0].shape[-2:]
        N_T = img_inputs[0].shape[1] // self.num_cam

        losses = dict()
        with torch.no_grad():
            if 'depth_preds' in kwargs:
                depth = kwargs['depth_preds']
            else:
                depth_out = self.estimate_depth(depth_input=kwargs['depth_img_inputs'], depth_size=(h // 2, w // 2))
                depth = depth_out['metric_depth']

        adj_metas = [img_inputs[8 + 4 * i] for i in range(N_T)] # lidarego2global, lidaregoprev2global_j
        semantic_results = self.pass_semantic_model(img_inputs[:7], depth, adj_metas)
        voxel_semantics = kwargs['voxel_semantics'].clone()
        # nuScenes
        mask_camera = kwargs['mask_camera'].clone()
        if N_T >= 3:
            torch.cuda.empty_cache()
        loss_occ = self.loss_occ(voxel_semantics, mask_camera, semantic_results, img_inputs)
        losses.update(loss_occ)

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

        h, w = img[0].shape[-2:]
        N_T = img[0].shape[1] // self.num_cam
        depth_out = self.estimate_depth(
            depth_input=kwargs['depth_img_inputs'][0],
            depth_size=(h // 2, w // 2),
        )
        depth = depth_out['metric_depth']
        
        adj_metas = [img[8 + 4 * i] for i in range(N_T)] # lidarego2global, lidaregoprev2global_j
        semantic_results = self.pass_semantic_model(img[:7], depth, adj_metas)

        sem_occ = semantic_results['sem_occ']
        bin_occ = semantic_results['bin_occ']

        sem_occ_max = torch.max(torch.softmax(sem_occ, dim=1), dim=1)
        sem_occ_cls, sem_occ_score = sem_occ_max.indices, sem_occ_max.values
        bin_occ_softmax = torch.softmax(bin_occ, dim=1)[:, 0]
        sel_tag = (sem_occ_score > 0.0) & (bin_occ_softmax > 0.5)
        free_idx = 17 if self.mode == "nuscenes" else 0
        occ_pred_cls = torch.where(sel_tag, sem_occ_cls, torch.ones_like(sem_occ_cls) * free_idx)
        occ_pred_cls = occ_pred_cls.permute(0, 3, 2, 1).contiguous()

        self.nonce += 1
        if self.retrieval:
            occ_res = self.compute_single_retrieval(
                bin_occ_softmax.squeeze(dim=0).permute(2, 1, 0).contiguous(),
                semantic_results['feat_occ'].squeeze(dim=0).permute(0, 3, 2, 1).contiguous(),
                kwargs['points_indices'][0].squeeze(dim=0), kwargs['retrieval_points'][0].squeeze(dim=0),
                kwargs['retrieval_anno'][0].squeeze(dim=0), img_metas[0]['retrieval_prompt']
            )
        else:
            occ_res = occ_pred_cls.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        return [occ_res]

    
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

    def pass_semantic_model(self, img_inputs, depth, adj_metas=None):
        imgs = img_inputs[0]
        img_metas = img_inputs[1:]
        output_dict = self.semantic_model(imgs, depth, img_metas, adj_metas)
        return output_dict

    def extract_feat(self, feat_inputs, depth=None, **kwargs):
        raise NotImplementedError

    def get_score_completion(self, predict, target, nonempty=None):
        """for scene completion, treat the task as two-classes problem, just empty or occupancy"""
        _bs = predict.shape[0]  # batch size
        # ---- ignore
        predict[target == 255] = 0
        target[target == 255] = 0
        # ---- flatten
        target = target.view(_bs, -1)  # (_bs, 129600)
        predict = predict.view(_bs, -1)  # (_bs, _C, 129600), 60*36*60=129600
        # ---- treat all non-empty object class as one category, set them to label 1
        b_pred = torch.zeros_like(predict)
        b_true = torch.zeros_like(target)
        b_pred[predict > 0] = 1
        b_true[target > 0] = 1

        tp_sum, fp_sum, fn_sum = 0, 0, 0
        for idx in range(_bs):
            y_true = b_true[idx, :]  # GT
            y_pred = b_pred[idx, :]
            if nonempty is not None:
                nonempty_idx = nonempty[idx, :].view(-1)
                y_true = y_true[nonempty_idx == 1]
                y_pred = y_pred[nonempty_idx == 1]

            tp = torch.sum((y_true == 1) & (y_pred == 1))
            fp = torch.sum((y_true != 1) & (y_pred == 1))
            fn = torch.sum((y_true == 1) & (y_pred != 1))
            tp_sum += tp
            fp_sum += fp
            fn_sum += fn

        return tp_sum, fp_sum, fn_sum

    def get_score_semantic_and_completion(self, predict, target, nonempty=None):
        _bs = predict.shape[0]  # batch size
        _C = 20  # _C = 12
        # ---- ignore
        predict[target == 255] = 0
        target[target == 255] = 0
        # ---- flatten
        target = target.view(_bs, -1)  # (_bs, 129600)
        predict = predict.view(_bs, -1)  # (_bs, 129600), 60*36*60=129600

        tp_sum = torch.zeros(_C).type_as(predict)
        fp_sum = torch.zeros(_C).type_as(predict)
        fn_sum = torch.zeros(_C).type_as(predict)

        for idx in range(_bs):
            y_true = target[idx]  # GT
            y_pred = predict[idx]

            if nonempty is not None:
                nonempty_idx = nonempty[idx, :].view(-1)
                valid_mask = (nonempty_idx == 1) & (y_true != 255)
                y_pred = y_pred[valid_mask]
                y_true = y_true[valid_mask]

            for j in range(_C):  # for each class
                tp = torch.sum((y_true == j) & (y_pred == j))
                fp = torch.sum((y_true != j) & (y_pred == j))
                fn = torch.sum((y_true == j) & (y_pred != j))
                tp_sum[j] += tp
                fp_sum[j] += fp
                fn_sum[j] += fn

        return tp_sum, fp_sum, fn_sum

    def compute_single_retrieval(self, occ_bin, occ_feat, indices, matching_points, anno, prompt=""):

        # Fetch the feat and bin stat from VEON output
        indices = indices.long()
        points_feat = occ_feat[:, indices[:, 0], indices[:, 1], indices[:, 2]]
        points_bin = occ_bin[indices[:, 0], indices[:, 1], indices[:, 2]]

        # Retrieval result in all points
        point_num = points_feat.shape[-1]

        retrieval_embedding = self.semantic_model.model.retrieval_embedding([prompt]).permute(1, 0).contiguous()
        cosine_score = F.cosine_similarity(points_feat,
                                           retrieval_embedding.repeat(1, point_num), dim=0)
        mAP = average_precision_score(anno.cpu().numpy(), cosine_score.cpu().numpy())
        print(mAP, prompt)

        # Retrieval result in visible points
        point_num_visible = matching_points.shape[0]
        points_feat_visible = points_feat[:, matching_points]
        points_bin_visible = points_bin[matching_points]
        anno_visible = anno[matching_points]
        cosine_score_visible = torch.cosine_similarity(points_feat_visible,
                                                       retrieval_embedding.repeat(1, point_num_visible), dim=0)
        mAP_visible = average_precision_score(anno_visible.cpu().numpy(), cosine_score_visible.cpu().numpy())
        ret = {"map": mAP, "map_visible": mAP_visible}
        return ret
