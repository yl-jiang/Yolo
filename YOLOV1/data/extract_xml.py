#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/31 16:28
# @Author  : jyl
# @File    : extract_xml.py
import os
import numpy as np
from PIL import Image
import random
from bs4 import BeautifulSoup
from V1.configs import opt
import lxml
from tqdm import tqdm


VOC_BBOX_LABEL_NAMES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                        'bus', 'car', 'cat', 'chair', 'cow',
                        'diningtable', 'dog', 'horse', 'motorbike', 'person',
                        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')


def read_image(path, dtype=np.float32, color=True):
    f = Image.open(path)
    try:
        if color:
            img = f.convert('RGB')
        else:
            img = f.convert('P')
        img = np.asarray(img, dtype=dtype)
    finally:
        if hasattr(f, 'close'):
            f.close()

    if img.ndim == 2:  # 灰度图片
        # reshape (H, W) -> (1, H, W)
        return img[np.newaxis]
    else:   # 彩色图片
        # transpose (H, W, C) -> (C, H, W)
        return img.transpose((2, 0, 1))


def parse_voc2012_xml(xml_file):
    one_file_bboxes = []
    one_file_labels = []
    bs = BeautifulSoup(open(xml_file), 'lxml')
    img_file_name = bs.find('filename').string
    for obj in bs.find_all('object'):
        diffcult = int(obj.find('difficult').string)
        if diffcult == 1:
            continue
        name = obj.find('name').string
        if name in VOC_BBOX_LABEL_NAMES:
            label = VOC_BBOX_LABEL_NAMES.index(name)
            bndbox_obj = obj.find('bndbox', recursive=False)
            y1 = int(float(bndbox_obj.find('ymax').string))
            x1 = int(float(bndbox_obj.find('xmax').string))
            y2 = int(float(bndbox_obj.find('ymin').string))
            x2 = int(float(bndbox_obj.find('xmin').string))
            one_file_bboxes.append([y1, x1, y2, x2])
            one_file_labels.append(label)

    return img_file_name, one_file_bboxes, one_file_labels


def parse_write_xml(voc_data_dir, write_train, write_test):
    AnnotationsPath = os.path.join(voc_data_dir, 'Annotations')
    xml_fils = os.listdir(AnnotationsPath)
    writer_train = open(write_train, 'a')
    writer_test = open(write_test, 'a')
    for f in tqdm(xml_fils):
        xml_path = os.path.join(AnnotationsPath, f)
        img_file_name, bboxes, labels = parse_voc2012_xml(xml_path)
        if len(labels) == 0:
            continue
        if random.random() >= 0.15:
            write_f = writer_train
        else:
            write_f = writer_test
        write_f.write(img_file_name)
        for bbox, label in zip(bboxes, labels):
            write_f.write(' '+str(bbox[0])+' '+str(bbox[1])+' '+str(bbox[2])+' '+str(bbox[3])+' '+str(label))
        write_f.write('\n')
    writer_train.close()
    writer_test.close()


if __name__ == '__main__':
    # test all data parser
    parse_write_xml(opt.voc_data_dir, opt.write_train, opt.write_test)
    # print(parse_voc2012_xml(r'D:\Data\ML\Object_Detection\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\Annotations\2007_002427.xml'))
    # parse_voc2012_xml(r'/home/dk/jyl/Object_Detection/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/Annotations/2008_000763.xml')
    # test one image parser
    # xml_path = r'/home/dk/jyl/Object_Detection/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/Annotations/2007_001284.xml'
    # parse_voc2012_xml(xml_path)
