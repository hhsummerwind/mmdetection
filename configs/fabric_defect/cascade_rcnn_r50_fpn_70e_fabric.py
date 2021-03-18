# -*- coding: utf-8 -*-
"""
Copyright 2021 huhui, Inc. All Rights Reserved.
@author: huhui
@software: PyCharm
@project: mmdetection	
@file: cascade_rcnn_r50_fpn_70e_fabric.py
@version: v1.0
@time: 2021/3/16 下午2:54
@setting: 
-------------------------------------------------
Description :
工程文件说明： 
"""

_base_ = [
    'cascade_rcnn_r50_fpn.py',
    'fabric_defect_detection.py',
    'schedule_70e.py', '../_base_/default_runtime.py'
]
