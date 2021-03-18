# *utf-8*
import os
import json
import numpy as np
import shutil
import pandas as pd
from tqdm import tqdm
from PIL import Image

panda_name2label = {
    'visible_body': 1, 'full_body': 2, 'head': 3, 'vehicle': 4
}


class Panda2COCO:

    def __init__(self, mode="train"):
        self.images = []
        self.annotations = []
        self.categories = []
        # self.img_id = 0
        self.ann_id = 0
        self.mode = mode

    def to_coco(self, person_anno_file, vechicle_ann_file):
        self._init_categories()
        person_anno_result = pd.read_json(open(person_anno_file, "r"))
        # name_list = anno_result["name"].unique()
        # for i, img_name in enumerate(tqdm(name_list)):
        # print(i, img_name)
        for path, value in person_anno_result.items():
            if int(path.split('_')[0]) <= 8:
                img_dir = img_dirs[0]
            else:
                img_dir = img_dirs[1]
            img_path = os.path.join(img_dir, path)
            image_id = value['image id']
            image_size = value['image size']
            h, w = image_size['height'], image_size['width']
            self.images.append(self._image(img_path, h, w, image_id))

            objects_list = value['objects list']
            for obj_dict in objects_list:
                category = obj_dict['category']
                if category != 'person':
                    continue
                rects = obj_dict['rects']

                head = rects['head']
                visible_body = rects['visible body']
                full_body = rects['full body']

                head_box = [
                    int(float(head['tl']['x']) * w),
                    int(float(head['tl']['y']) * h),
                    int(float(head['br']['x']) * w),
                    int(float(head['br']['y']) * h),
                ]
                visible_body_box = [
                    int(float(visible_body['tl']['x']) * w),
                    int(float(visible_body['tl']['y']) * h),
                    int(float(visible_body['br']['x']) * w),
                    int(float(visible_body['br']['y']) * h),
                ]
                full_body_box = [
                    int(float(full_body['tl']['x']) * w),
                    int(float(full_body['tl']['y']) * h),
                    int(float(full_body['br']['x']) * w),
                    int(float(full_body['br']['y']) * h),
                ]

                if head_box[1] > h and head_box[0] > w:
                    label = panda_name2label['head']
                    annotation = self._annotation(label, head_box, h, w)
                    if annotation is not None:
                        self.annotations.append(annotation)

                if bbox[1] >= h:
                    # print(bbox)
                    continue
                if bbox[0] >= w:
                    # print(bbox)
                    continue
                label = panda_name2label[defect_name]
                annotation = self._annotation(label, bbox, h, w)
                if annotation is not None:
                    self.annotations.append(annotation)
                self.ann_id += 1
            # self.img_id += 1
        instance = {}
        instance['info'] = 'fabric defect'
        instance['license'] = ['none']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    def _init_categories(self):
        # for v in range(1, 16):
        # print(v)
        # category = {}
        # category['id'] = v
        # category['name'] = str(v)
        # category['supercategory'] = 'defect_name'
        # self.categories.append(category)
        for k, v in panda_name2label.items():
            category = {}
            category['id'] = v
            category['name'] = k
            category['supercategory'] = 'panda_name'
            self.categories.append(category)

    def _image(self, path, h, w, img_id):
        image = {}
        image['height'] = h
        image['width'] = w
        image['id'] = img_id
        image['file_name'] = os.path.basename(path)
        return image

    def _annotation(self, label, bbox, h, w):
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        # area=abs(bbox[2]-bbox[0])*abs(bbox[3]-bbox[1])
        if area <= 0:
            print(bbox)
            return None
            # input()
        points = [[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]]
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = label
        annotation['segmentation'] = [np.asarray(points).flatten().tolist()]
        annotation['bbox'] = self._get_box(points, h, w)
        annotation['iscrowd'] = 0
        annotation['area'] = area
        return annotation

    def _get_box(self, points, img_h, img_w):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        '''coco,[x,y,w,h]'''
        w = max_x - min_x
        h = max_y - min_y
        if w > img_w:
            w = img_w
        if h > img_h:
            h = img_h
        return [min_x, min_y, w, h]

    def save_coco_json(self, instance, save_path):
        with open(save_path, 'w') as fp:
            json.dump(instance, fp, indent=1, separators=(',', ': '))


'''转换有瑕疵的样本为coco格式'''
# img_dir = "/data/datasets/天池广东2019布匹瑕疵检测/data/fabric/defect_Images"
img_dirs = ['/data/datasets/PANDA/panda_round1_train_202104_part1',
            '/data/datasets/PANDA/panda_round1_train_202104_part2']
anno_dir = "/data/datasets/天池广东2019布匹瑕疵检测/data/fabric/Annotations/anno_train_round2.json"
fabric2coco = Panda2COCO()
train_instance = fabric2coco.to_coco(anno_dir, img_dir)
fabric2coco.save_coco_json(train_instance, "/data/datasets/天池广东2019布匹瑕疵检测/data/fabric/annotations/"
                           + 'instances_{}.json'.format("train_20191004_mmd"))
