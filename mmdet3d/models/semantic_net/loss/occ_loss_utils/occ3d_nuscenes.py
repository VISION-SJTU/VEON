import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
# from mmcv.runner import BaseModule, force_fp32
from torch.cuda.amp import autocast

semantic_kitti_class_frequencies = np.array(
    [
        5.41773033e09,
        1.57835390e07,
        1.25136000e05,
        1.18809000e05,
        6.46799000e05,
        8.21951000e05,
        2.62978000e05,
        2.83696000e05,
        2.04750000e05,
        6.16887030e07,
        4.50296100e06,
        4.48836500e07,
        2.26992300e06,
        5.68402180e07,
        1.57196520e07,
        1.58442623e08,
        2.06162300e06,
        3.69705220e07,
        1.15198800e06,
        3.34146000e05,
    ]
)

kitti_class_names = [
    "empty",
    "car",
    "bicycle",
    "motorcycle",
    "truck",
    "other-vehicle",
    "person",
    "bicyclist",
    "motorcyclist",
    "road",
    "parking",
    "sidewalk",
    "other-ground",
    "building",
    "fence",
    "vegetation",
    "trunk",
    "terrain",
    "pole",
    "traffic-sign",
]



def inverse_sigmoid(x, sign='A'):
    x = x.to(torch.float32)
    while x >= 1-1e-5:
        x = x - 1e-5

    while x< 1e-5:
        x = x + 1e-5

    return -torch.log((1 / x) - 1)

def KL_sep(p, target):
    """
    KL divergence on nonzeros classes
    """
    nonzeros = target != 0
    nonzero_p = p[nonzeros]
    kl_term = F.kl_div(torch.log(nonzero_p), target[nonzeros], reduction="sum")
    return kl_term


def geo_scal_loss(pred, ssc_target, ignore_index=255, bg_idx=17):

    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)

    # Compute empty and nonempty probabilities
    empty_probs = pred[:, 1]
    nonempty_probs = pred[:, 0]

    # Remove unknown voxels
    mask = ssc_target != ignore_index
    nonempty_target = ssc_target != bg_idx
    nonempty_target = nonempty_target[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]

    eps = 1e-5
    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / (nonempty_probs.sum() + eps)
    recall = intersection / (nonempty_target.sum() + eps)
    spec = ((1 - nonempty_target) * empty_probs).sum() / ((1 - nonempty_target).sum() + eps)
    with autocast(False):
        return (
            F.binary_cross_entropy_with_logits(inverse_sigmoid(precision, 'A'), torch.ones_like(precision))
            + F.binary_cross_entropy_with_logits(inverse_sigmoid(recall, 'B'), torch.ones_like(recall))
            + F.binary_cross_entropy_with_logits(inverse_sigmoid(spec, 'C'), torch.ones_like(spec))
        )



def sem_scal_loss(pred, ssc_target, ignore_index=255):
    with autocast(False):
        # pred = F.softmax(pred_, dim=1)
        loss = 0
        count = 0
        mask = ssc_target != ignore_index
        n_classes = pred.shape[1]
        for i in range(n_classes):

            # Get probability of class i
            p = pred[:, i]  

            # Remove unknown voxels
            target_ori = ssc_target
            p = p[mask]
            target = ssc_target[mask]   

            completion_target = torch.ones_like(target)
            completion_target[target != i] = 0
            completion_target_ori = torch.ones_like(target_ori).float()
            completion_target_ori[target_ori != i] = 0
            if torch.sum(completion_target) > 0:
                count += 1.0
                nominator = torch.sum(p * completion_target)
                loss_class = 0
                if torch.sum(p) > 0:
                    precision = nominator / (torch.sum(p)+ 1e-5)
                    loss_precision = F.binary_cross_entropy_with_logits(
                            inverse_sigmoid(precision, 'D'), torch.ones_like(precision)
                        )
                    loss_class += loss_precision
                if torch.sum(completion_target) > 0:
                    recall = nominator / (torch.sum(completion_target) + 1e-5)
                    # loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))

                    loss_recall = F.binary_cross_entropy_with_logits(inverse_sigmoid(recall, 'E'), torch.ones_like(recall))
                    loss_class += loss_recall
                if torch.sum(1 - completion_target) > 0:
                    specificity = torch.sum((1 - p) * (1 - completion_target)) / (
                        torch.sum(1 - completion_target) +  1e-5
                    )

                    loss_specificity = F.binary_cross_entropy_with_logits(
                            inverse_sigmoid(specificity, 'F'), torch.ones_like(specificity)
                        )
                    loss_class += loss_specificity
                loss += loss_class
                # print(i, loss_class, loss_recall, loss_specificity)
        l = loss / count
        return l


def CE_ssc_loss(pred, target, class_weights=None, ignore_index=255):
    """
    :param: prediction: the predicted tensor, must be [BS, C, ...]
    """

    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=ignore_index, reduction="mean"
    )
    with autocast(False):
        loss = criterion(pred, target.long())

    return loss

def BCE_ssc_loss(pred, target, class_weights=None, ignore_index=255, pos_weight=1):
    pred = pred.permute(0, 2, 3, 4, 1).contiguous()
    pred = pred.reshape(-1, pred.shape[-1])
    target = target.reshape(-1)
    ignore_mask = target != ignore_index
    pred = pred[ignore_mask]
    target = target[ignore_mask]
    target = F.one_hot(target.long()).float()

    # loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none", pos_weight=class_weights)
    loss = F.binary_cross_entropy(pred, target, reduction="none")
    inst_num, class_num = target.shape

    loss = loss * class_weights[None, :]
    loss_tot = loss.sum() / inst_num * class_num

    return loss_tot


def vel_loss(pred, gt):
    with autocast(False):
        return F.l1_loss(pred, gt)


def BCE_BinOcc_Loss(pred, target, class_weights, ignore_index=255):
    """
    :param: prediction: the predicted tensor, must be [BS, 2, H, W, D]
    """
    target_bin = target.clone()
    target_bin[target < 17] = 0
    target_bin[target == 17] = 1
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=ignore_index, reduction="mean"
    )
    loss = criterion(pred, target_bin.long())

    return loss


def CE_SemOcc_Loss(pred, target, class_weights, ignore_index=255):
    """
    :param: prediction: the predicted tensor, must be [BS, C, H, W, D]
    C = 18 for 17 semantic classes and 1 free classes
    """
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=ignore_index, reduction="mean"
    )
    loss = criterion(pred, target.long())

    return loss


class Proj2Dto3DLoss(nn.Module):
    def __init__(self, grid_config=None, loss_det_weight=1.0, 
                 loss_soft_weight=1.0, ov_class_number=0, 
                 high_conf_thr=0.99, stage2_start=2, priority=None):
        super().__init__()
        self.grid_config = grid_config
        self.loss_det_weight = loss_det_weight
        self.loss_soft_weight = loss_soft_weight
        self.downsample = 1
        self.stage2_start = stage2_start
        self.high_conf_thr = high_conf_thr
        self.epoch = 0
        self.cosine = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.ov_class_number = ov_class_number
        self.priority = torch.tensor(priority)
        self.additional_cls_weights = torch.tensor(priority)
        
        print("Open-Vocabulary Class Number: {}".format(self.ov_class_number))
        print(f"Stage 2 start {self.stage2_start}, high conf thr {self.high_conf_thr}, ignore version!")


    def _merge_classes_prob(self, tensor, dim, class_reflection=None):
        dim_length = tensor.shape[dim]
        assert tensor.shape[dim] == len(class_reflection)
        merged = []
        left = 0
        while left < dim_length:
            right = left
            while right < dim_length - 1 and class_reflection[left] == class_reflection[right + 1]:
                right += 1
            sel_indices = list(range(left, right + 1))
            sel_indices = torch.tensor(sel_indices, device=tensor.device)
            cur_tensor = torch.index_select(tensor, dim=dim, index=sel_indices)
            cur_tensor = torch.max(cur_tensor, dim=dim, keepdim=True).values
            merged.append(cur_tensor)
            left = right + 1
        merged_tensor = torch.cat(merged, dim=dim)
        return merged_tensor

    def _onehot_restricted_class(self, class_prob, gt_semantic, class_reflection=None):
        restricted_max_indices = torch.zeros(class_prob.shape[1], dtype=torch.long, device=gt_semantic.device)
        default_indices = torch.zeros(class_prob.shape[1], dtype=torch.long, device=gt_semantic.device)
        restricted_max_probs = torch.zeros(class_prob.shape[1], device=gt_semantic.device)
        left = 0
        class_id = 0
        while left < len(class_reflection):
            right = left
            while right < len(class_reflection) - 1 and class_reflection[left] == class_reflection[right + 1]:
                right += 1
            sel_indices = list(range(left, right + 1))
            sel_indices = torch.tensor(sel_indices, device=gt_semantic.device)
            sel_probs = torch.index_select(class_prob, dim=0, index=sel_indices)
            max_meta = torch.max(sel_probs, dim=0)
            max_indices = max_meta.indices + left
            max_probs = max_meta.values
            partial_sel_idx = gt_semantic == class_id
            restricted_max_probs[partial_sel_idx] = max_probs[partial_sel_idx]
            restricted_max_indices[partial_sel_idx] = max_indices[partial_sel_idx]
            default_indices[partial_sel_idx] = class_id
            left = right + 1
            class_id += 1
        # merged_tensor = torch.cat(merged, dim=dim)
        return restricted_max_indices, restricted_max_probs, default_indices

    @torch.no_grad()
    def sample_imgfeat_from2d(self, map_2d, points_img, spatial_size, sem_valid):
        height, width = spatial_size
        coor = points_img[:, :2]
        depth = points_img[:, 2]
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] <= width - 1) & (
            coor[:, 1] >= 0) & (coor[:, 1] <= height - 1) & (
                depth < self.grid_config['depth'][1]) & (
                    depth >= self.grid_config['depth'][0]) & sem_valid
        # abq = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
        #         coor[:, 1] >= 0) & (coor[:, 1] < height)
        # print(kept1.shape, kept1.sum(), abq.sum())
        coor = coor[kept1]
        coor_raw = coor.clone()
        coor[:, 0] = (coor[:, 0] / ((width - 1) / 2)) - 1
        coor[:, 1] = (coor[:, 1] / ((height - 1) / 2)) - 1
        coor_raw = coor_raw[:, [1, 0]]
        # coor = coor[:, [1, 0]]
        # coor = torch.rand(coor.shape, device=coor.device) * 2 - 1

        map_2d = map_2d[None, :]
        coor_2d = coor[None, None, :]
        sampled_map = F.grid_sample(map_2d, coor_2d, mode='bilinear', align_corners=False)

        return kept1, sampled_map.squeeze(2).squeeze(0), coor_raw


    def visualize_projected_points(self, image, coords_raw, vis_folder="loader/proj_test", tag=""):

        def denormalize_img(x):
            """Reverses the imagenet normalization applied to the input.

            Args:
                x (torch.Tensor - shape(3,H,W)): input tensor

            Returns:
                torch.Tensor - shape(3,H,W): Denormalized input
            """
            mean = torch.Tensor([122.7709, 116.7460, 104.0937]).view(3, 1, 1).to(x.device)
            std = torch.Tensor([68.5005, 66.6322, 70.3232]).view(3, 1, 1).to(x.device)
            denorm_tensor = x * std + mean
            return denorm_tensor

        import os
        if not os.path.exists(vis_folder):
            os.makedirs(vis_folder, exist_ok=True)
        denorm_image = denormalize_img(image)
        # denorm_img = (denorm_image.cpu().numpy()).astype(np.uint8)
        # img = Image.fromarray(np.transpose(denorm_img, (1, 2, 0))[:, :, ::-1])
        # img.save(os.path.join(vis_folder, tag + "_gt.png"))

        coords_raw = torch.round(coords_raw).to(torch.long)
        coords_raw[:, 0] = torch.clip(coords_raw[:, 0], min=0, max=image.shape[-2] - 1)
        coords_raw[:, 1] = torch.clip(coords_raw[:, 1], min=0, max=image.shape[-1] - 1)
        denorm_image[:, coords_raw[:, 0], coords_raw[:, 1]] = 255
        denorm_img = (denorm_image.cpu().numpy()).astype(np.uint8)
        img = Image.fromarray(np.transpose(denorm_img, (1, 2, 0))[:, :, ::-1])
        img.save(os.path.join(vis_folder, tag + "_proj.png"))

    def forward(self, pred_feat_occ, sem_seg_2d, sem_embed_2d, 
                img_inputs, prev_img_inputs=None, voxel_semantics=None,
                sem_seg_2d_prev=None, sem_embed_2d_prev=None, 
                class_reflection=None, ov_classifier_weight=None, class_num=18):
        # Remove free class, index 17
        assert class_num == 18
        class_num -= 1

        sem_seg_2d = sem_seg_2d.detach()
        sem_embed_2d = sem_embed_2d.detach()
        if sem_embed_2d_prev is not None:
            sem_seg_2d_prev = sem_seg_2d_prev.detach()
            sem_embed_2d_prev = sem_embed_2d_prev.detach()
        self.additional_cls_weights = self.additional_cls_weights.to(pred_feat_occ.device)
        self.priority = self.priority.to(pred_feat_occ.device)

        B, C, H, W, Z = pred_feat_occ.shape
        pred_feat_occ = pred_feat_occ.reshape(B, C, -1).permute(0, 2, 1)
        coord_x, coord_y, coord_z = torch.meshgrid(torch.arange(H).to(pred_feat_occ.device),
                                                   torch.arange(W).to(pred_feat_occ.device),
                                                   torch.arange(Z).to(pred_feat_occ.device))
        coord_x = coord_x * self.grid_config['x'][2] + (self.grid_config['x'][0] + self.grid_config['x'][2] / 2)
        coord_y = coord_y * self.grid_config['y'][2] + (self.grid_config['y'][0] + self.grid_config['y'][2] / 2)
        coord_z = coord_z * self.grid_config['z'][2] + (self.grid_config['z'][0] + self.grid_config['z'][2] / 2)
        coord_xyz = torch.stack([coord_x, coord_y, coord_z], dim=-1)
        coord_xyz_flatten = coord_xyz.reshape(-1, 3)


        # ================================ Fetch parameters ===============================
        imgs, _, _, intrins = img_inputs[:4]
        post_rots, post_trans, bda = img_inputs[4:7]
        lidar2lidarego, lidarego2global, cam2camego, camego2global = img_inputs[7:]

        spatial_size = (imgs.shape[-2], imgs.shape[-1])
        camera_num = intrins.shape[1]

        loss_det, loss_soft = 0, 0
        for b in range(B):
            # Calculate 2D to 3D features and class maps
            loss_cur_dets, loss_cur_softs = [], []
            det_nums, soft_nums = [], []
            for cid in range(camera_num):

                # Coordinate transform
                cam2img = np.eye(4, dtype=np.float32)
                cam2img = torch.from_numpy(cam2img).to(coord_xyz.device)
                cam2img[:3, :3] = intrins[b, cid]

                # ========================== Transform for the current frame ==============================
                lidarego2cam = torch.inverse(camego2global[b, cid].matmul(cam2camego[b, cid])).matmul(
                    lidarego2global[b, cid])
                lidarego2img = cam2img.matmul(lidarego2cam)
                points_img = coord_xyz_flatten[:, :3].matmul(
                    lidarego2img[:3, :3].T) + lidarego2img[:3, 3].unsqueeze(0)
                points_img = torch.cat(
                    [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]], 1)
                points_img = points_img.matmul(
                    post_rots[b, cid].T) + post_trans[b, cid:cid + 1, :]

                voxel_semantics_rsp = voxel_semantics[b].reshape(-1)
                # Here sem_valid use gt_semantics < class_num, it's because in Occ3D-NuScenes, index = 17 means free
                sem_valid = (voxel_semantics_rsp < class_num) & (voxel_semantics_rsp >= 0)
                valid_lifted_3d, class_lifted_3d, coor_2d_raw = \
                    self.sample_imgfeat_from2d(sem_seg_2d[b, cid], points_img, spatial_size, sem_valid)
                gt_semantics = voxel_semantics_rsp[valid_lifted_3d]
                pred_feat_occ_cur = pred_feat_occ[b][valid_lifted_3d].contiguous()

                class_prob_3d = torch.softmax(class_lifted_3d, dim=0)
                class_indices_3d = torch.max(class_lifted_3d, dim=0).indices
                restricted_max_indices, restricted_max_probs, default_indices = \
                    self._onehot_restricted_class(class_prob_3d, gt_semantics, class_reflection)
                # Stat for correct classes
                class_map_3d_ds = self._merge_classes_prob(tensor=class_lifted_3d, dim=0,
                                                           class_reflection=class_reflection)
                class_indices_3d_ds = torch.max(class_map_3d_ds, dim=0).indices

                # ========================== Calculate Align Loss ===============================
                sel_soft = torch.logical_or(class_indices_3d_ds == gt_semantics, gt_semantics >= (class_num - self.ov_class_number))
                if b == B - 1 and cid == camera_num - 1 and sel_soft.shape[0] > 0:
                    sel_soft[0] = True
                # Loss for soft classes
                sel_det = ~sel_soft
                if b == B - 1 and cid == camera_num - 1 and sel_det.shape[0] > 0:
                    sel_det[0] = True

                # For the lifted class
                class_map_3d_onehot = F.one_hot(restricted_max_indices, num_classes=ov_classifier_weight.shape[0])
                feat_labels_3d_det = torch.einsum(
                    'nc,cd->nd', class_map_3d_onehot.float(), ov_classifier_weight).contiguous()
                num_det_cam = sel_det.sum().item()
                if num_det_cam:
                    # Loss without class-aware re-weighting
                    # loss_det_cam = 1 - self.cosine(feat_labels_3d_det[sel_det], feat_occ_cur[sel_det]).mean()

                    # Loss with class-aware re-weighting
                    loss_det_cam_each = 1 - self.cosine(feat_labels_3d_det[sel_det], pred_feat_occ_cur[sel_det])
                    one_hot_weight = F.one_hot(gt_semantics[sel_det].long(), num_classes=class_num)
                    each_class_sum = one_hot_weight.sum(dim=0)
                    class_exist_tag = each_class_sum > 0
                    each_class_weight = (1 / each_class_sum[class_exist_tag])
                    each_instance_weight = torch.matmul(one_hot_weight[:, class_exist_tag].float(), each_class_weight)
                    loss_det_cam = (loss_det_cam_each * each_instance_weight).sum() / \
                                   (self.additional_cls_weights[class_exist_tag]).sum()
                    del_weight = 0 if class_num == self.ov_class_number else 1
                    loss_cur_dets.append(loss_det_cam * del_weight)
                    det_nums.append(num_det_cam)

                class_map_nr_onehot = F.one_hot(class_indices_3d, num_classes=ov_classifier_weight.shape[0])
                feat_labels_3d_nr = torch.einsum(
                    'nc,cd->nd', class_map_nr_onehot.float(), ov_classifier_weight).contiguous()

                # Priority-Concerned Loss Item Ignorance
                if self.epoch >= self.stage2_start:
                    with torch.no_grad():
                        pred_probs_3d = torch.einsum(
                            'nc,dc->nd', pred_feat_occ_cur.detach().float(), ov_classifier_weight[:-1, :])
                        pred_indices_3d = torch.max(pred_probs_3d, dim=1).indices
                        pred_class_onehot = F.one_hot(pred_indices_3d, num_classes=ov_classifier_weight.shape[0] - 1)
                        pred_pseudo_gt_feat = torch.einsum(
                            'nc,cd->nd', pred_class_onehot.float(), ov_classifier_weight[:-1, :]).contiguous()
                        cosine_distance = self.cosine(pred_feat_occ_cur, pred_pseudo_gt_feat)
                        pred_class_ds = self._merge_classes_prob(tensor=pred_probs_3d, dim=1,
                                                                class_reflection=class_reflection)
                        pred_indices_3d_ds = torch.max(pred_class_ds, dim=1).indices
                        pred_priority = self.priority[pred_indices_3d_ds]
                        class_3d_priority = self.priority[class_indices_3d_ds]
                        high_conf_tag = torch.logical_and(cosine_distance >= self.high_conf_thr, pred_priority > class_3d_priority)
                        
                        # ====================== Key Ignore Operation =========================
                        sel_soft = sel_soft & (~high_conf_tag)

                # Soft cosine loss
                num_soft_cam = sel_soft.sum().item()
                if num_soft_cam:
                    # Loss without class-aware re-weighting
                    # loss_soft_cam = 1 - self.cosine(feat_map_3d[sel_soft], feat_occ_cur[sel_soft]).mean()

                    # Loss with class-aware re-weighting
                    loss_soft_cam_each = 1 - self.cosine(feat_labels_3d_nr[sel_soft], pred_feat_occ_cur[sel_soft])
                    one_hot_weight = F.one_hot(class_indices_3d_ds[sel_soft], num_classes=class_num)
                    each_class_sum = one_hot_weight.sum(dim=0)
                    class_exist_tag = each_class_sum > 0
                    each_class_weight = (1 / each_class_sum[class_exist_tag]) * self.additional_cls_weights[class_exist_tag]
                    each_instance_weight = torch.matmul(one_hot_weight[:, class_exist_tag].float(), each_class_weight)
                    loss_soft_cam = (loss_soft_cam_each * each_instance_weight).sum() / \
                                    (self.additional_cls_weights[class_exist_tag]).sum()

                    # loss_soft_cam = loss_soft_cam_each.mean()
                    loss_cur_softs.append(loss_soft_cam)
                    soft_nums.append(num_soft_cam)


            if len(loss_cur_dets) > 0:
                tot_det = max(1.0, sum(det_nums))
                loss_det_all_cams = sum(
                    ld * wd / tot_det for (ld, wd) in zip(loss_cur_dets, det_nums))
                loss_det = loss_det + loss_det_all_cams

            if len(loss_cur_softs) > 0:
                tot_soft = max(1.0, sum(soft_nums))
                loss_soft_all_cams = sum(ls * ws / tot_soft for (ls, ws) in zip(loss_cur_softs, soft_nums))
                loss_soft = loss_soft + loss_soft_all_cams


        return loss_det / B, loss_soft / B
