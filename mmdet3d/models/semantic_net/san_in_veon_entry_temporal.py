import random
from typing import List, Union, Dict, Tuple
import numpy as np

try:
    # ignore ShapelyDeprecationWarning from fvcore
    import warnings

    from shapely.errors import ShapelyDeprecationWarning

    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
except:
    pass
import os

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.visualizer import Visualizer, random_color
from mmdet3d.models.builder import NECKS
from PIL import Image
import torch.nn as nn

from .configs.san_config import add_san_config
from .vocabulary.nuscenes_vol import NUSCENES_CLASSES, NUSCENES_CLASSES_BRIEF
from .vocabulary.semkitti_vol import SEMKITTI_CLASSES_BRIEF
from .vocabulary.coco_vol import COCO_CATEGORIES
# from san.data.datasets.register_coco_stuff_164k import COCO_CATEGORIES

model_cfg = {
    "san_vit_b_16": {
        "config_file": "configs/san_clip_vit_res4_coco_temporal.yaml",
    },
    "san_vit_large_16": {
        "config_file": "configs/san_clip_vit_large_res4_coco_temporal.yaml",
    },
}

def setup(model_type: str, device=None):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_san_config(cfg)
    config_file = model_cfg[model_type]["config_file"]
    config_file_abs = os.path.join(os.path.dirname(os.path.realpath(__file__)), config_file)
    cfg.merge_from_file(config_file_abs)
    cfg.MODEL.DEVICE = device or "cuda" if torch.cuda.is_available() else "cpu"
    cfg.freeze()
    return cfg


@NECKS.register_module()
class SANInVeonEntryTemporal(nn.Module):
    def __init__(self, model_type: str, vocabulary: str, occ_size: Tuple[int],
                 num_frame: int = 1, num_camera: int = 6, num_temporal: int = 1,
                 grid_config: Dict = None, ds_feat: List[int] = None, ):
        """
        Args:
            model_type (str): the config file path
        """

        super().__init__()
        cfg = setup(model_type)
        cfg.defrost()
        cfg.MODEL.PROPAGATION_NETWORK.TEMPORAL_FUSION=num_temporal
        cfg.freeze()
        self.model = DefaultTrainer.build_model(cfg)
        self.grid_config, self.ds_feat = grid_config, ds_feat
        self.prepare_vocabulary(augment_vocabulary=vocabulary)
        self.num_frame, self.num_camera, self.occ_size = num_frame, num_camera, occ_size

    def prepare_vocabulary(
        self,
        vocabulary: List[str] = [],
        augment_vocabulary: str = "nuscenes_default",
    ) -> None:
        """
        Predict the segmentation result.
        Args:
            image_data_or_path (Union[Image.Image, str]): the input image or the image path
            vocabulary (List[str]): the vocabulary used for the segmentation
            augment_vocabulary (bool): whether to augment the vocabulary
            output_file (str): the output file path
        Returns:
            Union[dict, None]: the segmentation result
        """

        vocabulary = list(set([v.lower().strip() for v in vocabulary]))
        # Remove invalid vocabulary
        vocabulary = [v for v in vocabulary if v != ""]
        detailed_description = [v for v in vocabulary if v != ""]
        class_reflection = [i for i in range(len(vocabulary))]

        if augment_vocabulary in ["nuscenes_default", "nuscenes_brief", "semkitti_brief"]:
            vocabulary, detailed_description, class_reflection = \
                self._add_vocabulary_nuscenes(vocabulary, detailed_description, class_reflection, augment_vocabulary)
        elif augment_vocabulary == 'coco_default':
            vocabulary, detailed_description, class_reflection = \
                self._add_vocabulary_coco(vocabulary, detailed_description, class_reflection, augment_vocabulary)
        print("vocabulary: ", vocabulary)
        print("detailed description: ", detailed_description)
        assert len(vocabulary) == len(detailed_description)
        self.mode = "nuscenes" if "nuscenes" in augment_vocabulary else "semkitti"
        self.default_vocabulary = vocabulary
        self.detailed_description = detailed_description
        self.class_reflection = class_reflection

        # Prepare vocabulary
        self.model.prepare_vocabulary(detailed_description)
        # Prepare parameters for temporal alignment
        self.model.prepare_temporal_align(self.grid_config, self.ds_feat)

    def prepare_lss(
            self,
            lss_module: nn.Module,
    ):
        self.model.prepare_lss(lss_module, self.num_frame, self.num_camera, self.occ_size)

    def split_image_style_tensors(self, tensor, n_cam=6):
        if isinstance(tensor, list):
            tensor = tensor[-1]
        b = tensor.shape[0]
        reshaped_tensor = tensor.reshape(b, n_cam, -1, *tensor.shape[2:])
        N_T = reshaped_tensor.shape[2]
        return reshaped_tensor[:, :, 0], [reshaped_tensor[:, :, ti] for ti in range(1, N_T)]

    def forward(
        self,
        image: torch.Tensor,
        depth: torch.Tensor,
        img_metas: List,
        adj_metas: List = [],
        vocabulary: List[str] = [],
        vis: bool = False,
    ) -> Dict:
        """
        Predict the segmentation result.
        Args:
            image_data_or_path (Union[Image.Image, str]): the input image or the image path
            vocabulary (List[str]): the vocabulary used for the segmentation
            augment_vocabulary (bool): whether to augment the vocabulary
            output_file (str): the output file path
        Returns:
            Union[dict, None]: the segmentation result
        """
        vocabulary_infer = self.default_vocabulary if len(vocabulary) == 0 else vocabulary
        result = self.model(image, depth, img_metas, adj_metas)
        
        sem_occ, ov_classifier_weight = self._merge_classes_prob(
            result['sem_occ'], dim=1, weight=result['ov_classifier_weight'])
        
        res = dict(class_reflection=self.class_reflection,
                    ov_classifier_weight=ov_classifier_weight,
                    sem_seg_ds=result['sem_seg_ds'],
                    # sem_seg=result['sem_seg'],
                    sem_embed_ds=result['sem_embed_ds'],
                    # sem_embed=result["sem_embed"],
                    clip_feat=result['clip_feat'],
                    feat_occ=result['feat_occ'],
                    bin_occ=result['bin_occ'],
                    sem_occ_raw=result['sem_occ'],
                    sem_occ=sem_occ)

        return res

    def visualize(
        self,
        image: Image.Image,
        sem_seg: np.ndarray,
        vocabulary: List[str],
        output_file: str = None,
        mode: str = "overlay",
        vis_folder: str = "loader/open_vocab",
    ) -> Union[Image.Image, None]:
        """
        Visualize the segmentation result.
        Args:
            image (Image.Image): the input image
            sem_seg (np.ndarray): the segmentation result
            vocabulary (List[str]): the vocabulary used for the segmentation
            output_file (str): the output file path
            mode (str): the visualization mode, can be "overlay" or "mask"
        Returns:
            Image.Image: the visualization result. If output_file is not None, return None.
        """
        # add temporary metadata
        # set numpy seed to make sure the colors are the same
        np.random.seed(0)
        colors = [random_color(rgb=True, maximum=255) for _ in range(len(vocabulary))]
        MetadataCatalog.get("_temp").set(stuff_classes=self.default_vocabulary, stuff_colors=colors)
        metadata = MetadataCatalog.get("_temp")

        def denormalize_img(x):
            """Reverses the imagenet normalization applied to the input.

            Args:
                x (torch.Tensor - shape(3,H,W)): input tensor

            Returns:
                torch.Tensor - shape(3,H,W): Denormalized input
            """
            # mmlab denormalize
            mean = torch.Tensor([123.675, 116.28, 103.53]).view(3, 1, 1).to(x.device)
            std = torch.Tensor([58.395, 57.12, 57.375]).view(3, 1, 1).to(x.device)
            # clipsan denormalize
            # mean = torch.Tensor([122.7709, 116.7460, 104.0937]).view(3, 1, 1).to(x.device)
            # std = torch.Tensor([68.5005, 66.6322, 70.3232]).view(3, 1, 1).to(x.device)
            denorm_tensor = x * std + mean
            denorm_img = (denorm_tensor.cpu().numpy()).astype(np.uint8)
            img = Image.fromarray(np.transpose(denorm_img, (1, 2, 0))[:, :, ::-1])
            return img

        image = denormalize_img(image)
        if mode == "overlay":
            v = Visualizer(image, metadata)
            v = v.draw_sem_seg(sem_seg, area_threshold=0).get_image()
            v = Image.fromarray(v)
        else:
            v = np.zeros((image.size[1], image.size[0], 3), dtype=np.uint8)
            labels, areas = np.unique(sem_seg, return_counts=True)
            sorted_idxs = np.argsort(-areas).tolist()
            labels = labels[sorted_idxs]
            for label in filter(lambda l: l < len(metadata.stuff_classes), labels):
                v[sem_seg == label] = metadata.stuff_colors[label]
            v = Image.fromarray(v)

        # Remove temporary metadata
        MetadataCatalog.remove("_temp")
        if not os.path.exists(vis_folder):
            os.makedirs(vis_folder, exist_ok=True)
        if output_file is None:
            return v
        v.save(os.path.join(vis_folder, output_file))
        image.save(os.path.join(vis_folder, output_file.replace('output', 'input')))
        print(f"saved to {output_file}")

    def _add_vocabulary_nuscenes(self, vocabulary: List[str], detailed_description: List[str], class_reflection: List[int],
                                 augment_vocabulary: str) -> (List[str], List[str]):
        selected_class = {"nuscenes_default": NUSCENES_CLASSES,
                          "nuscenes_brief": NUSCENES_CLASSES_BRIEF,
                          "semkitti_brief": SEMKITTI_CLASSES_BRIEF}
        default_voc = []
        detailed_voc = []
        class_refl = []
        start_class = 0 if len(class_reflection) == 0 else class_reflection[-1] + 1
        for i, category_info in enumerate(selected_class[augment_vocabulary]):
            brief_items = [obj[0] for obj in category_info["detailed_items"]]
            detailed_items = [obj[0] if len(obj) == 1 else obj[0] + ", in detail \'" + obj[1] + "\'" for obj in category_info["detailed_items"]]
            class_items = [start_class] * len(brief_items)
            start_class += 1
            default_voc.extend(brief_items)
            detailed_voc.extend(detailed_items)
            class_refl.extend(class_items)
        return vocabulary + [c for c in default_voc if c not in vocabulary], \
               detailed_description + [dc for c, dc in zip(default_voc, detailed_voc) if c not in vocabulary], \
               class_reflection + [cr for c, cr in zip(default_voc, class_refl) if c not in vocabulary]

    def _add_vocabulary_coco(self, vocabulary: List[str], detailed_description: List[str], class_reflection: List[int],
                               augment_vocabulary: str) -> (List[str], List[str]):
        selected_class = {"coco_default": COCO_CATEGORIES}
        default_voc = [c["name"] for c in selected_class[augment_vocabulary]]
        append_voc = [c for c in default_voc if c not in vocabulary]
        start_class = 0 if len(class_reflection) == 0 else class_reflection[-1] + 1
        return vocabulary + append_voc, detailed_description + append_voc, \
               class_reflection + list(range(start_class, start_class + len(append_voc)))

    def _merge_classes_prob(self, tensor, dim, weight):
        dim_length = tensor.shape[dim]
        assert tensor.shape[dim] == len(self.class_reflection) + 1
        merged = []
        left = 0
        while left < dim_length:
            right = left
            while right < dim_length - 2 and self.class_reflection[left] == self.class_reflection[right + 1]:
                right += 1
            sel_indices = list(range(left, right + 1))
            sel_indices = torch.tensor(sel_indices, device=tensor.device)
            cur_tensor = torch.index_select(tensor, dim=dim, index=sel_indices)
            cur_tensor = torch.max(cur_tensor, dim=dim, keepdim=True).values
            merged.append(cur_tensor)
            left = right + 1

        # For semkitti, free index should be in 0
        weight = weight.clone()
        if self.mode == "semkitti":
            merged[0] = merged.pop(-1)
            weight[0, :] = weight[-1, :]
            weight = weight[:-1]

        merged_tensor = torch.cat(merged, dim=dim)
        return merged_tensor, weight

    def _postprocess(
        self, result: torch.Tensor, ori_vocabulary: List[str]
    ) -> np.ndarray:
        """
        Postprocess the segmentation result.
        Args:
            result (torch.Tensor): the segmentation result
            ori_vocabulary (List[str]): the original vocabulary used for the segmentation
        Returns:
            np.ndarray: the postprocessed segmentation result
        """
        result = result.argmax(dim=2).cpu().numpy()  # (H, W)
        if len(ori_vocabulary) == 0:
            return result
        result[result >= len(ori_vocabulary)] = len(ori_vocabulary)
        return result

