#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/8/3 下午10:09
# @Author  : jyl
# @File    : InferenceFunctions.py
import torch
from V1.configs import opt
from V1.bbox_utils import iou
from collections import defaultdict


def inference(bbox_pre, conf_pre, class_pre):
    """
    :param class_pre:
        [batch_size, 7, 7, 20]
    :param conf_pre:
        [batch_size, 7, 7, 2]
    :param bbox_pre:
        [batch_size, 7, 7, 2, 4] / [ymax, xmax, ymin, xmin]
    :return:
    """
    batch_conf_list = []
    batch_bbox_list = []
    batch_cls_list = []
    for i in range(bbox_pre.size(0)):
        batch_conf_list.append(conf_pre[i])
        batch_cls_list.append(class_pre[i])
        batch_bbox_list.append(bbox_pre[i])
    # filter these bbox whom confidence is litter than opt.iuo_threshold
    with torch.no_grad():
        for j in range(bbox_pre.size(0)):
            conf_cls = batch_conf_list[j].unsqueeze_(dim=-1) * batch_cls_list[j].unsqueeze_(dim=-2)  # [7, 7, 2, 20]
            cls = torch.argmax(conf_cls, dim=-1, keepdim=False).reshape(-1, 1)  # [7*7*2, 1]
            conf = torch.max(conf_cls, dim=-1, keepdim=False)[0].reshape(-1, 1)  # [7*7*2, 1]
            bbox = batch_bbox_list[j].reshape(-1, 4)  # [7*7*2, 4]

            # 只选择保留bbox的面积和置信度都符合要求的预测
            conf_mask = conf >= opt.conf_threshold
            conf_mask.squeeze_(dim=-1)  # [7*7*2, ]
            bbox_area = torch.prod(bbox[:, [0, 1]] - bbox[:, [2, 3]], dim=1)
            area_mask = bbox_area > opt.bbox_area_threshold
            mask = conf_mask + area_mask
            mask = torch.where(mask == 2, torch.ones_like(conf_mask), torch.zeros_like(conf_mask))

            batch_conf_list[j] = conf[mask.byte()]
            batch_bbox_list[j] = bbox[mask.byte()]
            batch_cls_list[j] = cls[mask.byte()]

            assert batch_conf_list[j].size(0) == batch_bbox_list[j].size(0) \
                   and batch_bbox_list[j].size(0) == batch_cls_list[j].size(0)
        # do NMS for remaining bbox
        batch_nms_bbox, batch_nms_conf, batch_nms_cls = yolo_nms(batch_bbox_list, batch_conf_list, batch_cls_list)
        return batch_nms_bbox, batch_nms_conf, batch_nms_cls


def yolo_nms(batch_bbox_list, batch_conf_list, batch_cls_list):
    batch_nms_bbox = defaultdict(list)
    batch_nms_conf = defaultdict(list)
    batch_nms_cls = defaultdict(list)
    for i in range(len(batch_bbox_list)):
        while True:
            keep_dict, remain_bbox, remain_conf, remain_cls = common_nms(batch_bbox_list[i], batch_conf_list[i],
                                                                         batch_cls_list[i])
            batch_nms_bbox[i].append(keep_dict['bbox'])
            batch_nms_conf[i].append(keep_dict['conf'])
            batch_nms_cls[i].append(keep_dict['cls'])
            batch_bbox_list[i] = remain_bbox
            batch_conf_list[i] = remain_conf
            batch_cls_list[i] = remain_cls
            if remain_bbox is None or len(remain_bbox) == 0:
                break
    return batch_nms_bbox, batch_nms_conf, batch_nms_cls


def common_nms(bboxes, confs, cls):
    """
    :param bboxes:
        shape:[N, 4]
    :param confs:
        shape:[N, 1]
    :param cls:
        shape:[N, 1]
    :return:
    """
    # 对某张图片进行iou过滤后若不存在bbox，则返回bbox为空，cls为-1
    if bboxes.size(0) == 0:
        keep_dict = {'bbox': [], 'conf': [0.], 'cls': [-1]}
        return keep_dict, None, None, None
    if bboxes.size(0) == 1:
        keep_dict = {'bbox': bboxes.squeeze(dim=0), 'conf': confs[0], 'cls': cls[0]}
        return keep_dict, None, None, None

    sort_mask = torch.argsort(confs.squeeze(dim=-1), descending=True)
    bbox_sorted = bboxes[sort_mask]
    confs_sorted = confs[sort_mask]
    cls_sorted = cls[sort_mask]

    max_conf_bbox = bbox_sorted[0].repeat(bboxes.size(0), 1)
    iou_output = iou(max_conf_bbox, bbox_sorted)
    remain_mask = iou_output <= opt.iou_threshold
    remain_mask = torch.tensor(remain_mask).byte()

    remain_bbox = bbox_sorted[remain_mask]
    remain_conf = confs_sorted[remain_mask]
    remain_cls = cls_sorted[remain_mask]
    keep_index = sort_mask[0]
    keep_dict = {'bbox': bboxes[keep_index], 'conf': confs[keep_index], 'cls': cls[keep_index]}

    return keep_dict, remain_bbox, remain_conf, remain_cls


if __name__ == '__main__':
    bboxes = [torch.tensor([[11, 11, 0, 0], [4, 6, 0, 0], [5, 5, 0, 0], [7, 7, 0, 0], [10, 10, 0, 0]])]
    confs = [torch.tensor([1, 0.7, 0.5, 0.2, 0.8])]
    cls = [torch.tensor([1, 2, 3, 4, 5])]
    batch_after_nms_bbox, batch_after_nms_conf, batch_after_nms_cls = yolo_nms(bboxes, confs, cls)
    print(batch_after_nms_bbox)
    print(batch_after_nms_conf)
    print(batch_after_nms_cls)
    # print(remainder_cls)
