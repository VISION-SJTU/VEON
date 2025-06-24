# Copyright (c) Phigent Robotics. All rights reserved.


_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams': 6,
    'input_size': (512, 1408),
    'depth_input_size': (256, 704),
    'src_size': (900, 1600),

    # Augmentation
    # 'resize': (-0.06, 0.11),
    # 'rot': (-5.4, 5.4),
    # 'flip': True,
    'resize': (-0.00, 0.00),
    'rot': (-0.0, 0.0),
    'flip': False,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

# Model
grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 0.4],
    'depth': [1.0, 45.0, 0.5],
}

multi_adj_frame_id_cfg = (1, 1, 1)
num_classes = 18
depth_pred_home="data/nuscenes/depth_cache/depth_dav2"

model = dict(
    type='VeonDepthCache',
    align_after_view_transfromation=False,
    num_adj=len(range(*multi_adj_frame_id_cfg)),
    num_classes=num_classes,
    mode="nuscenes",
    depth_mode="depthanythingv2",
    depth_pred_home=depth_pred_home,
    img_view_transformer=dict(
        type='LSSViewTransformerRaw',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        sid=False,
        collapse_z=False,
        out_channels=256,
        downsample=16,
        mode="nuscenes",
        loss_depth_weight=0.05,
        ds_feat=[2, 2, 2], # [z, h, w]
    ),
    depth_estimator=dict(
        # DepthAnything-V2 configs
        type='DepthAnythingV2Adaptor',
        # # Small version
        # encoder='vits', 
        # features=64, 
        # out_channels=[48, 96, 192, 384],
        # # Base version
        # encoder='vitb', 
        # features=128, 
        # out_channels=[96, 192, 384, 768],
        # # Large version
        encoder='vitl', 
        features=256, 
        out_channels=[256, 512, 1024, 1024],
        max_depth=80.0,
        use_lora=True,
        lora_r=16,
    ),
    loss_occ=dict(
        type='OccLossFB',
        out_channel=18,
        empty_idx=17,
        ignore_idx=255,
        grid_config=grid_config,
        mode="nuscenes",
        high_conf_thr=0.99,
        priority=[1] * 17,
        ov_class_number=17,
    ),
    use_mask=True,
)

test_cfg = dict(depth_estimator=True)

# Data
dataset_type = 'NuScenesDatasetOccpancy'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.0,
    flip_dy_ratio=0.0)

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        sequential=True,
        data_config=data_config,
        use_depth_input=True,
        # img_norm_method="clipsan",
        depth_img_norm_method="depthanythingv2",
        use_depth_pred=False,
        depth_pred_home=None),
    dict(type='LoadOccGTFromFile'),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=True),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='PointToOccPseudoLabel', grid_config=grid_config),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_depth', 'voxel_semantics',
                                'mask_lidar', 'mask_camera', 'mask_pseudo', 
                                'depth_img_inputs', 'depth_preds'])
]

test_pipeline = [
    dict(
        type='PrepareImageInputs',
        data_config=data_config,
        sequential=False,
        use_depth_input=True,
        # img_norm_method="clipsan",
        depth_img_norm_method="depthanythingv2"),
    # TODO: Remove load gt occ here, it's only used for vis
    dict(type='LoadOccGTFromFile'),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(512, 1408),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs', 'depth_img_inputs',
                                         'voxel_semantics', 'mask_camera'])
        ])
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    classes=class_names,
    modality=input_modality,
    stereo=False,
    filter_empty_gt=False,
    img_info_prototype='bevdet4d',
    multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
)

test_data_config = dict(
    pipeline=test_pipeline,
    ann_file=data_root + 'bevdetv2-nuscenes_infos_val.pkl')

data = dict(
    samples_per_gpu=1,  # with 32 GPU
    workers_per_gpu=2,
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'bevdetv2-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=test_data_config,
    test=test_data_config)

for key in ['val', 'train', 'test']:
    data[key].update(share_data_config)

# Optimizer
find_unused_parameters = False
# TODO: revise back weight decay
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[100,])
runner = dict(type='EpochBasedRunner', max_epochs=24)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
    dict(
        type='SyncbnControlHook',
        syncbn_start_epoch=0,
    ),
]

load_from='ckpts/clipsan/SAN_ViT-L.pth'
revise_keys=[(r'^', 'semantic_model.model.')]
depth_load_from='ckpts/depth_pretrain/depthanythingv2_pretrain_large.pth'
depth_revise_keys=[]
depth_test_time=True
# fp16 = dict(loss_scale='dynamic')
