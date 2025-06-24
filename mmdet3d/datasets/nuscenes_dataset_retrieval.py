# Copyright (c) OpenMMLab. All rights reserved.
import os
import mmcv
import torch
import cv2
import numpy as np
from tqdm import tqdm

from .builder import DATASETS
from .nuscenes_dataset import NuScenesDataset
from .occ_metrics import Metric_mIoU, Metric_FScore
from prettytable import PrettyTable

colors_map = np.array(
    [
        [0,   0,   0, 255],  # 0 undefined
        [255, 158, 0, 255],  # 1 car  orange
        [0, 0, 230, 255],    # 2 pedestrian  Blue
        [47, 79, 79, 255],   # 3 sign  Darkslategrey
        [220, 20, 60, 255],  # 4 CYCLIST  Crimson
        [255, 69, 0, 255],   # 5 traffic_light  Orangered
        [255, 140, 0, 255],  # 6 pole  Darkorange
        [233, 150, 70, 255], # 7 construction_cone  Darksalmon
        [255, 61, 99, 255],  # 8 bicycle  Red
        [112, 128, 144, 255],# 9 motorcycle  Slategrey
        [222, 184, 135, 255],# 10 building Burlywood
        [0, 175, 0, 255],    # 11 vegetation  Green
        [165, 42, 42, 255],  # 12 trunk  nuTonomy green
        [0, 207, 191, 255],  # 13 curb, road, lane_marker, other_ground
        [75, 0, 75, 255], # 14 walkable, sidewalk
        [255, 0, 0, 255], # 15 unobserved
        [0, 0, 0, 0],  # 16 undefined
        [0, 0, 0, 0],  # 16 undefined
    ])



@DATASETS.register_module()
class NuScenesDatasetRetrieval(NuScenesDataset):

    def __init__(self, ret_split="val", num_adjacent=0, *args, **kwargs):
        super(NuScenesDatasetRetrieval, self).__init__(*args, **kwargs)
        self.num_adjacent = num_adjacent
        seqs = self.read_retrieval_split(split=ret_split)
        self.filter_sequences(seqs)
        print("Retrieval Loading Over.")

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        input_dict = super(NuScenesDatasetRetrieval, self).get_data_info(index)
        # standard protocol modified from SECOND.Pytorch
        if 'occ_path' in self.data_infos[index]:
            input_dict['occ_gt_path'] = self.data_infos[index]['occ_path']
        return input_dict

    def get_adj_info(self, info, index):
        return self.data_infos_adj[index]

    def read_retrieval_split(self, split="eval"):
        filename = os.path.join("data/nuscenes/retrieval_benchmark", "retrieval_anns_{}.csv".format(split))
        import csv
        seqs = []
        with open(filename, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')
            for row in spamreader:
                token, seq_split, anno, matching_points, prompt = row
                cur = {"token": token, "split": seq_split, "anno": anno,
                       "matching_points": matching_points, "prompt": prompt}
                seqs.append(cur)
        print("Totally {} scenes in split {}.".format(len(seqs), split))
        return seqs

    def filter_sequences(self, seqs):
        # valid_tokens = {instance['token']: instance for instance in seqs}
        filtered_data_infos = []
        filtered_adjacent_infos = []
        for seq in seqs:
            for i, data_info in enumerate(self.data_infos):
                if seq['token'] == data_info['token']:
                    data_info_copy = data_info.copy()
                    data_info_copy["retrieval_meta"] = seq
                    filtered_data_infos.append(data_info_copy)
                    adj_infos = []
                    if 'scene_token' in data_info:
                        scene_token = data_info['scene_token']
                        for gap in range(1, self.num_adjacent + 1):
                            j = max(0, i - gap)
                            if self.data_infos[j]['scene_token'] != scene_token:
                                j = i
                            adj_infos.append(self.data_infos[j].copy())
                    else:
                        for _ in range(self.num_adjacent):
                            adj_infos.append(self.data_infos[i].copy())
                    filtered_adjacent_infos.append(adj_infos)
                    break

        # for data_info in self.data_infos:
        #     if data_info['token'] in valid_tokens:
        #         data_info["retrieval_meta"] = valid_tokens[data_info['token']]
        #         filtered_data_infos.append(data_info)
        self.data_infos = filtered_data_infos
        self.data_infos_adj = filtered_adjacent_infos
        print("After filtering, {} scenes remains.".format(len(self.data_infos)))

    def evaluate(self, occ_results, runner=None, show_dir=None, **eval_kwargs):

        print('\nStarting Evaluation...')
        map, map_visible = [], []
        for index, occ_pred in enumerate(tqdm(occ_results)):
            map.append(occ_pred['map'] * 100)
            map_visible.append(occ_pred['map_visible'] * 100)

        mAP_avg = "{:.1f}".format(np.mean(np.array(map)))
        visible_mAP_avg = "{:.1f}".format(np.mean(np.array(map_visible)))

        x = PrettyTable()
        x.field_names=['method', "mAP", "mAP visible"]
        x.title = 'POP3D Retrieval Benchmark'
        x.add_row(["Ours", mAP_avg, visible_mAP_avg])

        return x

