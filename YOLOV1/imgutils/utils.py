#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/31 17:58
# @Author  : jyl
# @File    : imgutils.py
import random
import numpy as np
import cv2


def resize_bbox(bbox, in_size, out_size):
    # bbox:(1,4)
    # 因为对图片预处理时对图片进行了缩放，那么之前人工标注的BoundingBox也必须按照等比例缩放
    y_scale = float(out_size[0] / in_size[0])
    x_scale = float(out_size[1] / in_size[1])

    bbox[:, 0] = bbox[:, 0] * y_scale
    bbox[:, 1] = bbox[:, 1] * x_scale
    bbox[:, 2] = bbox[:, 2] * y_scale
    bbox[:, 3] = bbox[:, 3] * x_scale
    return bbox


def random_flip(img, horizontal_flip=False, vertical_flip=False, return_flip=False):
    import random
    horizontal, vertical = False, False
    if horizontal_flip:
        horizontal = random.choice([False, True])
    if vertical_flip:
        vertical = random.choice([False, True])

    if horizontal:
        img = img[:, ::-1, :]
    if vertical:
        img = img[:, :, ::-1]

    if return_flip:
        return img, {'horizontal': horizontal, 'vertical': vertical}
    else:
        return img


def flip_bbox(bbox, img_size, horizontal=False, vertical=False):
    h, w = img_size
    bbox = bbox.copy()  # 深复制
    if horizontal:  # 水平翻转横坐标不变
        y1 = h - bbox[:, 0]
        y2 = h - bbox[:, 2]
        bbox[:, 0] = y1
        bbox[:, 2] = y2
    if vertical:  # 垂直翻转纵坐标不变
        x1 = w - bbox[:, 1]
        x2 = w - bbox[:, 3]
        bbox[:, 1] = x1
        bbox[:, 3] = x2

    return bbox


def encode_bbox(bbox, label, img_size, S, B, C):
    """

    :param bbox: (1,4)/ndarray
    :param label: (1,)/ndarray
    :param img_size: (1,2)/[448,448]
    :param S: 7
    :param B: 2
    :param C: 20
    :return:
    """
    import numpy as np
    target = np.zeros((S, S, B * 5 + C))
    center_y = (bbox[:, 0] + bbox[:, 2]) / 2
    center_x = (bbox[:, 1] + bbox[:, 3]) / 2
    h = bbox[:, 2] - bbox[:, 0]
    w = bbox[:, 3] - bbox[:, 1]
    cell_size = img_size[0] / S

    # gt_bbox中心点相对于img左上角的偏移量
    y = center_y / img_size[0]
    x = center_x / img_size[0]

    # gt_bbox中心点在单元格中的坐标
    cell_y = center_y / cell_size
    cell_x = center_x / cell_size

    # 找到gt_bbox中心点落于哪个单元格
    row = int(cell_y)
    col = int(cell_x)

    # gt_bbox中心点相对于所在单元格左上角的偏移量
    cell_scale_y = cell_y - row
    cell_scale_x = cell_x - col

    h_in_img = h / img_size[0]
    w_in_img = w / img_size[1]

    trans_coor = [y, x, h_in_img, w_in_img]

    target[row, col, [4, 9]] = [1., 1.]
    target[row, col, :4] = trans_coor
    target[row, col, 5:9] = trans_coor
    target[row, col, 9+label[0]] = 1.

    y_true = {"target": target,  # (7,7,30)/ndarray
              "gt_cell_index": [row, col],  # gt_bbox中心点在单元格中的位置
              "gt_center": [center_y, center_x],  # gt_bbox中心点坐标
              "gt_hw": [h, w],  # gt_bbox的长宽
              "gt_center_scale": [cell_scale_y, cell_scale_x]  # gt_bbox中心点坐标与所在单元格左上角的偏移量
              }
    return y_true


def images_db(file_path):
    f = open(file_path, 'r')
    file_names = list()
    bboxes = list()
    labels = list()
    for line in f.readlines():
        splits = line.strip().split()
        file_names.append(splits[0])
        num_obj = int(len(splits[1:]) / 5)
        bbox = []
        label = []
        for i in range(num_obj):
            bbox.append([int(splits[5*i+1]), int(splits[5*i+2]), int(splits[5*i+3]), int(splits[5*i+4])])
            label.append(int(splits[5*i+5]))
        bboxes.append(bbox)
        labels.append(label)
    return file_names, np.array(bboxes), np.array(labels)


def BGR2RGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)