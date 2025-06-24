# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/FB-BEV/blob/main/LICENSE

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.core import reduce_mean
from mmdet.models import HEADS, LOSSES
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
# from mmdet3d.models.semantic_net.loss.occ_loss_utils import lovasz_softmax, CustomFocalLoss
from mmdet3d.models.semantic_net.loss.occ_loss_utils import nusc_class_frequencies, nusc_class_names
from mmdet3d.models.semantic_net.loss.occ_loss_utils \
    import BCE_BinOcc_Loss, CE_SemOcc_Loss, Proj2Dto3DLoss, geo_scal_loss
from mmcv.runner import BaseModule, force_fp32
from torch.cuda.amp import autocast
from mmdet3d.models import builder

@LOSSES.register_module()
class OccLossFB(BaseModule):
    def __init__(
        self,
        out_channel=18,
        loss_weight_cfg=None,
        empty_idx=17,
        ignore_idx=255,
        balance_cls_weight=True,
        use_focal_loss=False,
        use_dice_loss=False,
        grid_config=None,
        mode="nuscenes",
        high_conf_thr=0.985, 
        stage2_start=2, 
        priority=None,
        ov_class_number=17,
    ):
        super(OccLossFB, self).__init__()

        self.use_focal_loss = use_focal_loss
        if self.use_focal_loss:
            self.focal_loss = builder.build_loss(dict(type='CustomFocalLoss'))

        if loss_weight_cfg is None:
            self.loss_weight_cfg = {
                "loss_2d_pixel_align_weight": 1.0,
                "loss_voxel_ce_weight": 1.5,
                "loss_voxel_sem_scal_weight": 0.25,
                "loss_voxel_geo_scal_weight": 0.25,
                "loss_featalign_det_weight": 35.0,
                "loss_featalign_soft_weight": 25.0,
            }
        else:
            self.loss_weight_cfg = loss_weight_cfg
        
        # voxel losses
        self.loss_2d_pixel_align_weight = self.loss_weight_cfg.get('loss_2d_pixel_align_weight', 1.0)
        self.loss_voxel_ce_weight = self.loss_weight_cfg.get('loss_voxel_ce_weight', 1.0)
        self.loss_voxel_sem_scal_weight = self.loss_weight_cfg.get('loss_voxel_sem_scal_weight', 1.0)
        self.loss_voxel_geo_scal_weight = self.loss_weight_cfg.get('loss_voxel_geo_scal_weight', 1.0)
        self.loss_featalign_det_weight = self.loss_weight_cfg.get("loss_featalign_det_weight", 1.0)
        self.loss_featalign_soft_weight = self.loss_weight_cfg.get("loss_featalign_soft_weight", 1.0)
            
        # loss functions
        self.use_dice_loss = use_dice_loss
        if self.use_dice_loss:
            self.dice_loss = builder.build_loss(dict(type='DiceLoss', loss_weight=2))

        if balance_cls_weight:
            self.class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:out_channel] + 0.001))
        else:
            self.class_weights = torch.ones(out_channel) / out_channel
        self.out_channel = out_channel
        self.bin_class_weights = torch.ones(2)
        self.bin_class_weights[1] = 0.5
        self.class_names = nusc_class_names    
        self.empty_idx = empty_idx
        self.ignore_idx = ignore_idx
        self.high_conf_thr = high_conf_thr
        self.stage2_start= stage2_start
        self.priority = priority
        assert mode in ["semkitti", "nuscenes"]
        self.bin_occ_loss = BCE_BinOcc_Loss
        self.proj2dto3dloss = Proj2Dto3DLoss(ov_class_number=ov_class_number, grid_config=grid_config,
                                                high_conf_thr=high_conf_thr, stage2_start=stage2_start, 
                                                priority=priority)

        self.ov_class_number = ov_class_number

    @force_fp32()
    def forward(self, voxel_semantics, mask_camera, semantic_results, img_inputs, **kwargs):
        n_cam = 6
        prev_image_inputs = []
        if img_inputs[0].shape[1] > n_cam:
            curr_image, prev_image = self.split_image_style_tensors(img_inputs[0])
            curr_metas, prev_metas =  self.split_image_metas(img_inputs[1:6])
            curr_additions, prev_additions = img_inputs[6:11], img_inputs[6:7] + img_inputs[11:]
            curr_image_inputs = [curr_image] + curr_metas + curr_additions
            prev_image_inputs = [prev_image] + prev_metas + prev_additions 
        else:
            curr_image_inputs = img_inputs
        
        meta_info = dict(img_inputs=curr_image_inputs, prev_img_inputs=prev_image_inputs)
        for meta_key in ["sem_seg", "sem_seg_ds", "sem_embed", "sem_embed_ds",
                         "clip_feat", "class_reflection", "ov_classifier_weight"]:
            if meta_key in semantic_results:
                meta_info[meta_key] = semantic_results[meta_key]
                del semantic_results[meta_key]
            new_meta_key = meta_key + "_prev"
            if new_meta_key in semantic_results:
                meta_info[new_meta_key] = semantic_results[new_meta_key]
                del semantic_results[new_meta_key]            

        loss = self.loss(semantic_results=[semantic_results],
                         voxel_semantics=voxel_semantics,
                         mask_camera=mask_camera,
                         meta_info=meta_info, **kwargs)

        return loss

    @force_fp32() 
    def loss_voxel(self, semantic_results, target_voxels, meta_info, tag):

        sem_occ = semantic_results['sem_occ'].permute(0, 1, 4, 3, 2)
        bin_occ = semantic_results['bin_occ'].permute(0, 1, 4, 3, 2)
        feat_occ = semantic_results['feat_occ'].permute(0, 1, 4, 3, 2)

        loss_dict = {}

        loss_dict['loss_binocc_{}'.format(tag)] = \
            self.loss_voxel_ce_weight * self.bin_occ_loss(
                bin_occ, target_voxels, self.bin_class_weights.type_as(bin_occ), ignore_index=self.ignore_idx)
        loss_featalign_det, loss_featalign_soft = \
            self.proj2dto3dloss(feat_occ, 
                                    meta_info['sem_seg_ds'], 
                                    meta_info['sem_embed_ds'],
                                    meta_info['img_inputs'], 
                                    prev_img_inputs=meta_info['prev_img_inputs'],
                                    voxel_semantics=target_voxels,
                                    class_reflection=meta_info['class_reflection'],
                                    ov_classifier_weight=meta_info['ov_classifier_weight'],
                                    class_num=self.out_channel)

        # Losses in loss dict
        if self.ov_class_number != self.out_channel - 1:
            # Meaning that not all classes are open-vocabulary
            loss_dict['loss_featalign_det_{}'.format(tag)] = loss_featalign_det * self.loss_featalign_det_weight
        if self.ov_class_number != 0:
            loss_dict['loss_featalign_soft_{}'.format(tag)] = loss_featalign_soft * self.loss_featalign_soft_weight

        return loss_dict

    @force_fp32() 
    def loss(self, voxel_semantics, mask_camera, semantic_results, meta_info, **kwargs):
        loss_dict = {}
        # loss_dict.update(self.loss_2d_pixel_align(meta_info))
        # Conversion
        voxel_semantics[mask_camera == 0] = self.ignore_idx
        for index, semantic_result in enumerate(semantic_results):
            loss_dict.update(self.loss_voxel(semantic_result, voxel_semantics, meta_info, tag='c_{}'.format(index)))
        return loss_dict

    def loss_2d_pixel_align(self, meta_info):

        max_class_idxs = torch.argmax(meta_info['sem_seg_ds'], dim=2)
        ov_classifier_weight = meta_info['ov_classifier_weight'][:-1, :]
        onehot_weight = F.one_hot(max_class_idxs, num_classes=ov_classifier_weight.shape[0]).float()
        onehot_weight = onehot_weight.permute(0, 1, 4, 2, 3).contiguous()

        pseudo_cls_embeddings = torch.einsum('bnchw,cd->bndhw', onehot_weight, ov_classifier_weight)
        pred_embeddings_2d = meta_info['clip_feat']
        B, N, C, H, W = pseudo_cls_embeddings.shape
        B, N, C, h, w = pred_embeddings_2d.shape
        pseudo_cls_embeddings = pseudo_cls_embeddings.reshape(-1, C, H, W)
        pred_embeddings_2d = nn.functional.interpolate(
            pred_embeddings_2d.reshape(-1, C, h, w), size=(H, W))

        loss_2d_cosine = 1 - nn.functional.cosine_similarity(pseudo_cls_embeddings, pred_embeddings_2d, 1, 1e-6).mean()
        return {'loss_2d_cosine': loss_2d_cosine * self.loss_weight_cfg['loss_2d_pixel_align_weight']}

    def split_image_style_tensors(self, tensor, n_cam=6):
        B = tensor.shape[0]
        assert B == 1, "only support batch size = 1"
        reshaped_tensor = tensor.reshape(B, n_cam, -1, *tensor.shape[2:])
        return reshaped_tensor[:, :, 0], reshaped_tensor[:, :, 1]

    def split_image_metas(self, img_metas, n_cam=6):
        B = img_metas[0].shape[0]
        assert B == 1, "only support batch size = 1"
        image_metas_curr, image_metas_prev = [], []
        for i in range(5):
            reshaped_mat = img_metas[i].reshape(B, -1, n_cam, *img_metas[i].shape[2:])
            image_metas_curr.append(reshaped_mat[:, 0])
            image_metas_prev.append(reshaped_mat[:, 1])
        return image_metas_curr, image_metas_prev
