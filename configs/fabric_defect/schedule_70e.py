# -*- coding: utf-8 -*-
"""
Copyright 2021 huhui, Inc. All Rights Reserved.
@author: huhui
@software: PyCharm
@project: mmdetection	
@file: schedule_70e.py
@version: v1.0
@time: 2021/3/16 上午11:21
@setting: 
-------------------------------------------------
Description :
工程文件说明： 
"""


# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
# optimizer = dict(type='SGD', lr=0.00375, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[40, 60])
runner = dict(type='EpochBasedRunner', max_epochs=70)
