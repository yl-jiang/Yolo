#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/31 16:41
# @Author  : jyl
# @File    : data.py
import numpy as np
import cv2
import os
from V1.configs import opt
from V1.data.extract_xml import parse_write_xml
from V1.imgutils import CVTransform
from V1.imgutils import images_db
from V1.imgutils import BGR2RGB
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class VocDataset(Dataset):
    """
    :return
        img:(batch_size,448,448)/ndarray
        gt_bbox:(batch_size,1,4)/ndarray
        gt_label:(batch_size,1)/ndarray
        scale:(batch_size,1,2)/ndarray
        y_true['target']:(batch_size,7,7,30)/ndarray
    """

    def __init__(self, is_train=True, show_img=False):
        self.is_train = is_train
        self.show_img = show_img
        if not os.path.exists(opt.write_train):
            parse_write_xml(opt.voc_data_dir, opt.write_train, opt.write_test)
        if is_train:
            self.file_names, self.bboxes, self.labels = images_db(opt.write_train)
        else:
            self.file_names, self.bboxes, self.labels = images_db(opt.write_test)
        self._check_init(self.bboxes, self.labels)

    def __len__(self):
        return len(self.file_names)

    @staticmethod
    def _check_init(bbox, label):
        if len(bbox) == 0 or len(label) == 0:
            raise ValueError('Lading image wrong! Bbox and label should be not empty!')

    def __getitem__(self, index):
        file_name = self.file_names[index]
        img_bgr = cv2.imread(os.path.join(opt.voc_data_dir, 'JPEGImages', file_name))
        bboxes = np.copy(self.bboxes[index])
        labels = np.copy(self.labels[index])
        if self.is_train:
            img_trans = CVTransform(img_bgr, bboxes, labels)
            img_bgr, bboxes, labels = img_trans.img, img_trans.bboxes, img_trans.labels
        img_rgb = BGR2RGB(img_bgr)
        resize_img, resize_bboxes = self.resize_img_bbox(img_rgb, bboxes)
        self.resized_bboxes = resize_bboxes
        self.grid_idx, self.grid_labels, target = self.make_target(resize_bboxes, labels, opt.img_size, opt.img_size)

        if self.show_img:
            for i, bbox in enumerate(resize_bboxes):
                bbox = bbox.astype(np.uint32)
                cv2.rectangle(resize_img, (bbox[1], bbox[0]), (bbox[3], bbox[2]), (55, 255, 155), 1)
            fig = plt.figure(figsize=(28, 14))
            ax1 = fig.add_subplot(111)
            ax1.xaxis.set_major_locator(plt.MultipleLocator(64))
            ax1.yaxis.set_major_locator(plt.MultipleLocator(64))
            ax1.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.001')
            ax1.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.001')
            ax1.imshow(resize_img)
            plt.show()

        img = self.normailze(resize_img, opt.mean, opt.std)
        # target:[[x, y, w, h], ...]
        if self.is_train:
            return img, target
        else:
            return file_name, resize_img, img, resize_bboxes, labels

    def make_target(self, resized_bboxes, labels, img_h, img_w):
        grid_h, grid_w = img_h / opt.S, img_w / opt.S
        row_elements = opt.B * 5 + opt.C
        target = np.zeros((opt.S, opt.S, row_elements))
        center_wh = self.reset_bboxes(resized_bboxes)  # [center_x, center_y, w, h]
        # [[row_id, col_id], ...]
        grid_idx = np.floor(center_wh[:, [1, 0]] / [grid_h, grid_w]).astype(np.int16)
        grid_idx, grid_labels, center_wh = self.remove_duplicate(grid_idx, labels, center_wh)
        scaled_center_wh = center_wh / [img_w, img_h, img_w, img_h]
        coor_offsets = (center_wh[:, [1, 0]] - grid_idx * [grid_h, grid_w]) / [grid_h, grid_w]  # [center_y, center_x]

        for idx, coor, wh, label in zip(grid_idx, coor_offsets, scaled_center_wh[:, [2, 3]], grid_labels):
            for i in range(opt.B):
                # x,y
                target[idx[0], idx[1], [0 + 5 * i, 1 + 5 * i]] = [coor[1], coor[0]]
                # w,h
                target[idx[0], idx[1], [2 + 5 * i, 3 + 5 * i]] = [np.sqrt(wh[0]), np.sqrt(wh[1])]
                # confidence
                target[idx[0], idx[1], [4 + 5 * i]] = [1.]
            # label
            target[idx[0], idx[1], 10 + label] = 1.
        # target: [[x, y, sqrt(w), sqrt(h)], ...]
        return grid_idx, grid_labels, target

    @staticmethod
    def reset_bboxes(bboxes):
        new_bbox = np.zeros_like(bboxes)
        box_hw = bboxes[:, [0, 1]] - bboxes[:, [2, 3]]
        box_center = (bboxes[:, [2, 3]] + bboxes[:, [0, 1]]) / 2  # [center_y, center_x]
        new_bbox[:, [1, 0]] = box_center
        new_bbox[:, [3, 2]] = box_hw
        return new_bbox  # [center_x, center_y, w, h]

    @staticmethod
    def resize_img_bbox(img_rgb, bbox):
        resized_img = cv2.resize(img_rgb, (opt.img_size, opt.img_size))
        w_scale = opt.img_size / img_rgb.shape[1]
        h_scale = opt.img_size / img_rgb.shape[0]
        resized_bbox = np.ceil(bbox * [h_scale, w_scale, h_scale, w_scale])
        return resized_img, resized_bbox

    @staticmethod
    def normailze(img, mean, std):
        torch_normailze = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        img = torch_normailze(img)
        return img

    @staticmethod
    def remove_duplicate(bboxes, labels, center_wh):
        container = {}
        assert bboxes.shape[0] == len(labels)
        mark = 0
        index = 0
        remove_ids = []
        for key, value in zip(bboxes, labels):
            container.setdefault(tuple(key), value)
            if len(container.keys()) == mark:
                remove_ids.append(index)
            mark = len(container.keys())
            index += 1
        center_wh_clear = np.delete(center_wh, remove_ids, axis=0)
        return np.array([list(k) for k in container.keys()]), list(container.values()), center_wh_clear


def choose_test_data(num):
    testset = VocDataset(is_train=False)
    data_length = len(testset)
    chosen_imgs = np.random.randint(low=0, high=data_length, size=num)
    img_fname = []
    raw_img = []
    input_img = []
    gt_label = []
    gt_bbox = []
    for img_id in chosen_imgs:
        img_fname.append(testset[img_id][0])
        raw_img.append(testset[img_id][1])
        input_img.append(testset[img_id][2].numpy()[None, ...])
        gt_bbox.append(testset[img_id][3])
        gt_label.append(testset[img_id][4])
    input_img = np.concatenate(input_img, axis=0)
    return img_fname, raw_img, input_img, gt_bbox, gt_label


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import torch

    vd = VocDataset(show_img=True)
    img, tar = vd[10]
    print(vd.grid_idx)
    print(vd.grid_labels)

