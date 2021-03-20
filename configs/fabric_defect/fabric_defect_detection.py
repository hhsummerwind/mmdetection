# -*- coding: utf-8 -*-
"""
Copyright 2021 huhui, Inc. All Rights Reserved.
@author: huhui
@software: PyCharm
@project: mmdetection	
@file: fabric_defect_detection.py
@version: v1.0
@time: 2021/3/16 上午11:09
@setting: 
-------------------------------------------------
Description :
工程文件说明： 
"""

# dataset settings
dataset_type = 'FabricDataset'
data_root = '/data/datasets/天池广东2019布匹瑕疵检测/data/fabric/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Concat', template_path=data_root + 'template_Images/'),
    dict(
        type='Resize',
        img_scale=[(3400, 800), (3400, 1200)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Concat', template_path='/tcdata/guangdong1_round2_testB_20191024/'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(3400, 1100),
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train_20191004_mmd.json',
        img_prefix=data_root + 'defect_Images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train_20191004_mmd.json',
        img_prefix=data_root + 'defect_Images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train_20191004_mmd.json',
        img_prefix=data_root + 'defect_Images/',
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox'])
