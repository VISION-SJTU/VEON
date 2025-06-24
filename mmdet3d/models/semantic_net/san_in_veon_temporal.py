from typing import List, Tuple

import open_clip
import torch
import copy
from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.memory import retry_if_cuda_oom
from torch import nn
from torch.nn import functional as F

from .clip_utils import (
    ClipOutput,
    FeatureExtractor,
    LearnableBgOvClassifier,
    PredefinedOvClassifier,
    RecWithAttnbiasHead,
    get_predefined_templates,
)
from .side_adapter import build_side_adapter_network_in_veon, build_hsa_network
from .side_adapter.align_net_occ3d import AlignNetOcc3D


@META_ARCH_REGISTRY.register()
class SANInVeonTemporal(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        clip_visual_extractor: nn.Module,
        clip_rec_head: nn.Module,
        side_adapter_network: nn.Module,
        highres_side_adaptor_network: nn.Module,
        ov_classifier: PredefinedOvClassifier,
        occ_decoder: nn.Module,
        # criterion: SetCriterion,
        size_divisibility: int,
        asymetric_input: bool = True,
        clip_resolution: float = 0.5,
        sem_seg_postprocess_before_inference: bool = False,
    ):
        super().__init__()
        self.asymetric_input = asymetric_input
        self.clip_resolution = clip_resolution
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.size_divisibility = size_divisibility

        self.side_adapter_network = side_adapter_network
        self.highres_side_adaptor_network = highres_side_adaptor_network
        self.clip_visual_extractor = clip_visual_extractor
        self.clip_rec_head = clip_rec_head
        self.ov_classifier = ov_classifier
        self.ov_classifier_weight = None
        # self.criterion = criterion
        self.occ_decoder = occ_decoder

    @classmethod
    def from_config(cls, cfg, **kwargs):

        ## end of copy
        model, _, preprocess = open_clip.create_model_and_transforms(
            cfg.MODEL.SAN.CLIP_MODEL_NAME,
            pretrained=cfg.MODEL.SAN.CLIP_PRETRAINED_NAME,
        )
        ov_classifier = LearnableBgOvClassifier(
            model, templates=get_predefined_templates(cfg.MODEL.SAN.CLIP_TEMPLATE_SET)
        )

        clip_visual_extractor = FeatureExtractor(
            model.visual,
            last_layer_idx=cfg.MODEL.SAN.FEATURE_LAST_LAYER_IDX,
            # frozen_exclude=cfg.MODEL.SAN.CLIP_FROZEN_EXCLUDE,
        )
        clip_rec_head = RecWithAttnbiasHead(
            model.visual,
            first_layer_idx=cfg.MODEL.SAN.FEATURE_LAST_LAYER_IDX,
            frozen_exclude=cfg.MODEL.SAN.CLIP_DEEPER_FROZEN_EXCLUDE,
            cross_attn=cfg.MODEL.SAN.REC_CROSS_ATTN,
            sos_token_format=cfg.MODEL.SAN.SOS_TOKEN_FORMAT,
            sos_token_num=cfg.MODEL.SIDE_ADAPTER.NUM_QUERIES,
            downsample_method=cfg.MODEL.SAN.REC_DOWNSAMPLE_METHOD,
        )

        occ_decoder = AlignNetOcc3D(clip_dim=cfg.MODEL.HIGHRES_SIDE_ADAPTOR.CLIP_DIM,
                                    hsa_dim=cfg.MODEL.HIGHRES_SIDE_ADAPTOR.DIM,
                                    embed_dim=cfg.MODEL.PROPAGATION_NETWORK.DIM,
                                    layer_lifting_map=cfg.MODEL.PROPAGATION_NETWORK.LIFTING_LAYERS,
                                    clip_outdim=cfg.MODEL.PROPAGATION_NETWORK.CLIP_PROJ_DIM,
                                    fusion_type=cfg.MODEL.PROPAGATION_NETWORK.FUSION_TYPE,
                                    layer_depth=cfg.MODEL.PROPAGATION_NETWORK.LAYER_DEPTH,
                                    num_temporal=cfg.MODEL.PROPAGATION_NETWORK.TEMPORAL_FUSION)

        return {
            "clip_visual_extractor": clip_visual_extractor,
            "clip_rec_head": clip_rec_head,
            "side_adapter_network": build_side_adapter_network_in_veon(
                cfg, clip_visual_extractor.output_shapes
            ),
            'highres_side_adaptor_network': build_hsa_network(
                cfg, clip_visual_extractor.output_shapes
            ),
            "ov_classifier": ov_classifier,
            # 'criterion': criterion,
            "size_divisibility": cfg.MODEL.SAN.SIZE_DIVISIBILITY,
            "asymetric_input": cfg.MODEL.SAN.ASYMETRIC_INPUT,
            "clip_resolution": cfg.MODEL.SAN.CLIP_RESOLUTION,
            "sem_seg_postprocess_before_inference": cfg.MODEL.SAN.SEM_SEG_POSTPROCESS_BEFORE_INFERENCE,
            "occ_decoder": occ_decoder,
        }

    def forward(self, images, depth, img_metas, adj_metas):
        # get classifier weight for each dataset
        # !! Could be computed once and saved. It will run only once per dataset.
        B, N, C, H, W = images.shape
        images = images.view(B * N, C, H, W)

        clip_input = F.interpolate(
            images, scale_factor=self.clip_resolution, mode="bilinear"
        ) if self.asymetric_input else images

        with torch.no_grad():
            clip_image_features = self.clip_visual_extractor(clip_input)
            mask_preds, attn_biases, san_features = self.side_adapter_network(
                images, clip_image_features
            )
            # !! Could be optimized to run in parallel.
            mask_embs = [
                self.clip_rec_head(clip_image_features, attn_bias, normalize=True)
                for attn_bias in attn_biases
            ]  # [B,N,C]
            # clip_image_features = mask_embs[0][1]
            # mask_embs = [mask_emb[0] for mask_emb in mask_embs]

            mask_logits = [
                torch.einsum("bqc,nc->bqn", mask_emb, self.ov_classifier_weight)
                for mask_emb in mask_embs
            ]

        outputs = {
            'ov_classifier_weight': self.ov_classifier_weight,
        }

        # Split the features and outputs
        n_cam = 6
        depth = depth.view(B * N, *depth.shape[-2:])
        depth, depth_prevs = self.split_image_style_tensors(depth, n_cam=n_cam, batch=B)
        depth, depth_prevs = depth.reshape(B, -1, *depth.shape[-2:]), \
                            [dp.reshape(B, -1, *dp.shape[-2:]) for dp in depth_prevs]
        mask_logits, mask_logits_prevs = self.split_image_style_tensors(mask_logits[-1], n_cam=n_cam, batch=B)
        mask_preds, mask_preds_prevs = self.split_image_style_tensors(mask_preds[-1], n_cam=n_cam, batch=B)
        mask_embeds, mask_embeds_prevs = self.split_image_style_tensors(mask_embs[-1], n_cam=n_cam, batch=B)
        images, images_prevs = self.split_image_style_tensors(images, n_cam=n_cam, batch=B)
        img_metas, img_metas_prevs = self.split_image_metas(img_metas, n_cam=n_cam)
        clip_image_features, clip_image_features_prevs = self.split_clip_outputs(clip_image_features, n_cam=n_cam)

        occ_feat_prevs = []
        for tid, (depth_prev, mask_logits_prev, mask_preds_prev, mask_embeds_prev, images_prev, img_metas_prev, clip_image_features_prev) in \
                enumerate(zip(depth_prevs, mask_logits_prevs, mask_preds_prevs, mask_embeds_prevs, images_prevs, img_metas_prevs, clip_image_features_prevs)):
            with torch.no_grad():
                # Inference 2D, only for the previous image
                _, sem_embed_ds_prev = self.semantic_inference_2d_w_embed(mask_logits_prev, mask_embeds_prev, mask_preds_prev)

                # Prepare 3D early feature for previous frame
                offsets_prev, attns_prev, supp_prev = self.highres_side_adaptor_network(images_prev, clip_image_features_prev)
                clip_image_features_prev = self.clip_rec_head.update_remaining_clip_feats(clip_image_features_prev, offsets_prev, attns_prev)
                occ_feat_prev = self.occ_decoder.forward_early(sem_embed_ds_prev, clip_image_features_prev, [supp_prev], depth_prev, img_metas_prev)

                # Occ feat Transform
                adj_metas_prev = [adj_metas[0], adj_metas[1 + tid]]
                occ_feat_prev = self.align_after_lss(occ_feat_prev, adj_metas_prev)
                occ_feat_prevs.append(occ_feat_prev)

        # Inference OV seg in 2D, for the current image
        sem_seg_ds, sem_embed_ds = self.semantic_inference_2d_w_embed(mask_logits, mask_embeds, mask_preds)
        outputs["sem_seg_ds"] = sem_seg_ds
        outputs["sem_embed_ds"] = sem_embed_ds
        mask_preds = F.interpolate(
            mask_preds,
            size=(images.shape[-2], images.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        sem_seg = self.semantic_inference_2d(mask_logits, mask_preds)
        outputs["sem_seg"] = sem_seg

        # Prepare 3D early feature for the current frame
        offsets, attns, supp = self.highres_side_adaptor_network(images, clip_image_features)
        clip_image_features = self.clip_rec_head.update_remaining_clip_feats(clip_image_features, offsets, attns)
        outputs['clip_feat'] = clip_image_features['clip_feat_proj']

        # Inference 3D
        occ_preds = self.occ_decoder(sem_embed_ds, clip_image_features, [supp], 
                                     depth, img_metas, occ_feat_prevs)
        feat_occ = F.interpolate(
            occ_preds["feat_occ"],
            size=self.occ_size,
            mode="trilinear",
            align_corners=False,
        )
        bin_occ = F.interpolate(
            occ_preds['bin_occ'],
            size=self.occ_size,
            mode="trilinear",
            align_corners=False,
        )
        sem_occ = self.semantic_inference_3d(self.ov_classifier_weight, feat_occ)
        outputs["sem_occ"] = sem_occ
        outputs["bin_occ"] = bin_occ
        outputs["feat_occ"] = feat_occ

        reshape_keys = ['clip_feat', 'sem_seg_ds', 'sem_seg', "sem_embed_ds"]
        for key in reshape_keys:
            if key in outputs:
                outputs[key] = outputs[key].reshape(B, -1, *outputs[key].shape[1:])

        return outputs

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros(
                (gt_masks.shape[0], h_pad, w_pad),
                dtype=gt_masks.dtype,
                device=gt_masks.device,
            )
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets

    def semantic_inference_2d(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        # mask_pred = mask_pred.sigmoid()
        # mask_cls = F.softmax(mask_cls, dim=1)
        semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
        return semseg

    def semantic_inference_2d_w_embed(self, mask_cls, mask_embed, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        # mask_pred = mask_pred.sigmoid()
        # mask_cls = F.softmax(mask_cls, dim=1)
        semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
        semembed = torch.einsum("bqc,bqhw->bchw", mask_embed, mask_pred)
        return semseg, semembed

    def semantic_inference_3d(self, ov_classifier_weight, mask_pred):
        semocc = torch.einsum("qc,bczhw->bqzhw", ov_classifier_weight, mask_pred)
        return semocc

    def prepare_vocabulary(self, vocabulary):
        self.ov_classifier_weight = (
                self.ov_classifier.logit_scale.exp()
                * self.ov_classifier.get_classifier_by_vocabulary(vocabulary)
        )
        self.ov_classifier_weight = self.ov_classifier_weight.clone().detach()

    def retrieval_embedding(self, vocabulary):
        retrieval_embedding = (
                self.ov_classifier.logit_scale.exp()
                * self.ov_classifier.get_classifier_by_vocabulary(vocabulary, add_bg=False)
        ).clone()
        return retrieval_embedding

    def prepare_lss(self, lss_module, num_frame=1, num_camera=6, occ_size=(16, 200, 200)):
        self.occ_decoder.lss_view_transformer = lss_module
        self.occ_decoder.num_frame = num_frame
        self.occ_decoder.num_camera = num_camera
        self.occ_size = occ_size

    def split_image_metas(self, img_metas, n_cam=6):
        B = img_metas[0].shape[0]
        N_T = img_metas[0].shape[1] // n_cam
        image_metas_curr, image_metas_prevs = [], [[] for _ in range(N_T - 1)] 
        for i in range(5):
            reshaped_mat = img_metas[i].reshape(B, -1, n_cam, *img_metas[i].shape[2:])
            image_metas_curr.append(reshaped_mat[:, 0])
            for tid in range(N_T - 1):
                image_metas_prevs[tid].append(reshaped_mat[:, 1 + tid])
        image_metas_curr.append(img_metas[-1])
        for tid in range(N_T - 1):
            image_metas_prevs[tid].append(img_metas[-1])
        return image_metas_curr, image_metas_prevs

    def split_clip_outputs(self, clip_out, n_cam=6):
        B = clip_out['0_cls_token'].shape[0]
        N_T = clip_out['0_cls_token'].shape[1] // n_cam
        clip_out_curr = ClipOutput(spacial_shape=clip_out.spacial_shape)
        clip_out_prevs = [ClipOutput(spacial_shape=clip_out.spacial_shape) for tid in range(N_T - 1)]
        for key in clip_out.keys():
            tensor = clip_out[key]
            if isinstance(key, int):
                reshaped_tensor = tensor.reshape(B, n_cam, -1, *tensor.shape[1:])
                clip_out_curr[key] = reshaped_tensor[:, :, 0].reshape(B * n_cam, *tensor.shape[1:])
                for tid in range(N_T - 1):
                    clip_out_prevs[tid][key] = reshaped_tensor[:, :, 1 + tid].reshape(B * n_cam, *tensor.shape[1:])
            else:
                reshaped_tensor = tensor.reshape(B, n_cam, -1, *tensor.shape[2:])
                clip_out_curr[key] = reshaped_tensor[:, :, 0]
                for tid in range(N_T - 1):
                    clip_out_prevs[tid][key] = reshaped_tensor[:, :, 1 + tid]

        return clip_out_curr, clip_out_prevs
    
    def split_image_style_tensors(self, tensor, n_cam=6, batch=1):
        reshaped_tensor = tensor.reshape(batch, n_cam, -1, *tensor.shape[1:])
        N_T = reshaped_tensor.shape[2]
        return reshaped_tensor[:, :, 0].reshape(batch * n_cam, *tensor.shape[1:]), \
               [reshaped_tensor[:, :, tid + 1].reshape(batch * n_cam, *tensor.shape[1:]) for tid in range(N_T - 1)]

    def prepare_temporal_align(self, grid_config, ds_feat):
        self.grid_config = grid_config
        self.ds_feat = ds_feat

    def align_after_lss(self, occ_feat, adj_metas):
        grid_config = copy.deepcopy(self.grid_config)
        lss_feat_ds = self.ds_feat
        for si, s in enumerate(['z', 'y', 'x']):
            grid_config[s][2] *= lss_feat_ds[si]

        B, C, Z, W, H = occ_feat.shape
        coord_x, coord_y, coord_z = torch.meshgrid(torch.arange(H).to(occ_feat.device),
                                                   torch.arange(W).to(occ_feat.device),
                                                   torch.arange(Z).to(occ_feat.device))
        coord_x = coord_x * grid_config['x'][2] + (grid_config['x'][0] + grid_config['x'][2] / 2)
        coord_y = coord_y * grid_config['y'][2] + (grid_config['y'][0] + grid_config['y'][2] / 2)
        coord_z = coord_z * grid_config['z'][2] + (grid_config['z'][0] + grid_config['z'][2] / 2)
        coord_xyz = torch.stack([coord_x, coord_y, coord_z], dim=-1)
        coord_xyz_flatten = coord_xyz.reshape(-1, 3)

        lidarego2global, lidaregoprev2global = adj_metas
        points_prevs = []
        for b in range(B):
            lidarego2lidaregoprev = torch.inverse(lidaregoprev2global[b, 0]).matmul(lidarego2global[b, 0])
            points_prev = coord_xyz_flatten[:, :3].matmul(
                        lidarego2lidaregoprev[:3, :3].T) + lidarego2lidaregoprev[:3, 3].unsqueeze(0)
            # (B, 3, Z, H, W)
            points_prev = points_prev.reshape(H, W, Z, 3).permute(2, 1, 0, 3)
            points_prevs.append(points_prev)
            # print(f"Mean coord xyz gap: {(points_prev.reshape(-1, 3) - coord_xyz_flatten).mean(0)}.")
        
        points_prevs = torch.stack(points_prevs, dim=0)
        singular = torch.clone(coord_xyz[0, 0, 0]).detach()
        scale = torch.clone(coord_xyz[-1, -1, -1]).detach() - singular
        points_prevs = (points_prevs - singular) / scale
        points_prevs = points_prevs * 2 - 1

        sampled_occ_feat = F.grid_sample(
            occ_feat,  
            points_prevs,
            align_corners=True,
            mode='bilinear',
            padding_mode='zeros'
        )
        return sampled_occ_feat
