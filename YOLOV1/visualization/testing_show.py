#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/8/17 上午11:12
# @Author  : jyl
# @File    : testing_show.py
import matplotlib.pyplot as plt
import numpy as np
from V1.configs import opt
import torch
import logging.config
import logging


VOC_BBOX_LABEL_NAMES = opt.VOC_BBOX_LABEL_NAMES + tuple(['BG'])
logging.config.fileConfig(opt.log_config)
logger = logging.getLogger('YoloV1Logger')


def vis_image(img, ax=None):
    assert img.shape[-1] == 3
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.imshow(img.astype(np.uint8))
    return ax


def show_test(visualization_dict):
    with torch.no_grad():
        batch_after_nms_bbox = visualization_dict['predict_bbox']  # [ymax, xmax, ymin, xmin]
        batch_after_nms_conf = visualization_dict['predict_conf']
        batch_after_nms_cls = visualization_dict['predict_class']
        GT_img = visualization_dict['GT_img']
        GT_img_name = visualization_dict['GT_fname']
        resized_img = visualization_dict['resized_img']
        shape_scale = visualization_dict['shape_scale']

        for i in range(len(GT_img)):
            bbox_scale = torch.from_numpy(np.tile(shape_scale[i], 2)).float().to(opt.device)
            ax = vis_image(GT_img[i])
            for k, box in enumerate(batch_after_nms_bbox[i]):
                try:
                    batch_after_nms_bbox[i][k] = box * bbox_scale
                except:
                    pass
            one_img_bboxes = batch_after_nms_bbox[i]
            one_img_confs = batch_after_nms_conf[i]
            one_img_classes = batch_after_nms_cls[i]
            ax = draw_predict_bbox(ax, one_img_bboxes, one_img_confs, one_img_classes, GT_img_name[i])
            plt.savefig(opt.img_save_path + '/{}.jpg'.format(GT_img_name[i]))
        plt.close('all')


def draw_predict_bbox(ax, one_img_bboxes, one_img_confs, one_img_classes, img_id):
    for j in range(len(one_img_bboxes)):
        if len(one_img_bboxes[j]) == 0:
            ax.text(0, 0, VOC_BBOX_LABEL_NAMES[-1], style='italic',
                    bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 1})
            continue
        # xy:左上角坐标；width：框的宽度；height：框的高度
        xy = (one_img_bboxes[j][3].item(), one_img_bboxes[j][2].item())
        width = one_img_bboxes[j][1].item() - one_img_bboxes[j][3].item()
        heigth = one_img_bboxes[j][0].item() - one_img_bboxes[j][2].item()
        label = opt.VOC_BBOX_LABEL_NAMES[one_img_classes[j]]
        score = one_img_confs[j].item()
        ax.add_patch(plt.Rectangle(xy, width, heigth, fill=False, edgecolor='green', linewidth=1.5))
        caption = [label, '%.3f' % score]
        ax.text(xy[0], xy[1], s=':'.join(caption), style='italic', bbox={'facecolor': 'green', 'alpha': 0.5, 'pad': 1})
        logger.info(f'{img_id} [ymin, xmin, ymax, xmax] = [{xy[1]}, {xy[0]}, {xy[1]+heigth}, {xy[0]+width}]')
    return ax


def draw_GT(ax, GT_bbox, GT_label):
    for j in range(len(GT_bbox)):
        # xy:左下角坐标；width：框的宽度；height：框的高度
        xy = (GT_bbox[j][3], GT_bbox[j][2])
        width = GT_bbox[j][1] - GT_bbox[j][3]
        heigth = GT_bbox[j][0] - GT_bbox[j][2]
        label = opt.VOC_BBOX_LABEL_NAMES[GT_label[j]]
        ax.add_patch(plt.Rectangle(xy, width, heigth, fill=False, edgecolor='red', linewidth=1.5))
        ax.text(xy[0], xy[1], label, style='italic', bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 1})
    return ax


