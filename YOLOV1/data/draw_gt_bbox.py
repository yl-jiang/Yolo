#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/8/8 上午10:03
# @Author  : jyl
# @File    : draw_gt_bbox.py
from V1.imgutils.utils import images_db
from tqdm import tqdm
from V1.configs import opt
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_bbox_in_img():
    file_names, bboxes, labels = images_db(opt.write_train)
    for i, img_id in tqdm(enumerate(file_names)):
        fname = os.path.join(opt.voc_data_dir, 'JPEGImages', img_id)
        img = Image.open(fname)
        assert img.mode == 'RGB'
        fig = plt.figure()
        # add_subplot为面向对象编程, plt.subplot为面向过程编程
        ax = fig.add_subplot(111)
        ax.imshow(img)
        for bbox, label in zip(bboxes[i], labels[i]):
            xy = (bbox[3], bbox[2])
            width = bbox[1] - bbox[3]
            height = bbox[0] - bbox[2]
            ax.add_patch(patches.Rectangle(xy, width, height, fill=False, facecolor='red', linewidth=1.5, color='red'))
            ax.text(xy[0], xy[1], s=opt.VOC_BBOX_LABEL_NAMES[label], style='italic', bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 2})
        plt.savefig('./vis_bbox/%s' % img_id)
        plt.close('all')


if __name__ == '__main__':
    draw_bbox_in_img()