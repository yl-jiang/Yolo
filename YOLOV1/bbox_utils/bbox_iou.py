#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/10 17:26
# @Author  : jyl
# @File    : bbox_iou.py
import numpy as np
import torch


def bbox_iou(gt_bbox, pre_bbox):
    """
    :param gt_bbox:
        [[ymax,xmax,ymin,xmin], ...]
        shape：[2m, 4]
    :param pre_bbox:
         [[ymax,xmax,ymin,xmin], ...]
         shape:[2m, 4]
    :return:
        [a, b, c,...]
    """
    with torch.no_grad():
        if gt_bbox.shape != pre_bbox.shape:
            raise ValueError("target_bbox and predic_bbox's shape must be same!")
        if isinstance(gt_bbox, torch.Tensor):
            gt_bbox = gt_bbox.cpu().detach().numpy()
        if isinstance(pre_bbox, torch.Tensor):
            pre_bbox = pre_bbox.cpu().detach().numpy()
        tl = np.maximum(gt_bbox[:, 2:], pre_bbox[:, 2:])  # (len(gt_bbox),2)
        br = np.minimum(gt_bbox[:, :2], pre_bbox[:, :2])  # (len(gt_bbox),2)
        area_i = np.prod(br - tl, axis=1) * (tl < br).all(axis=1)  # (len(gt_bbox),)
        area_gt = np.prod(gt_bbox[:, :2] - gt_bbox[:, 2:], axis=1)  # (len(gt_bbox),)
        # 确保gt_bbox的前后两个bbox值相同
        assert (area_gt[::2] == area_gt[1::2]).all()
        area_pre = np.prod(pre_bbox[:, :2] - pre_bbox[:, 2:], axis=1)  # (len(gt_bbox),)
        area_pre_mask = area_pre > 0
        area_pre = area_pre * area_pre_mask.astype(np.uint8)
        iou = area_i / (area_pre + area_gt - area_i)  # (len(gt_bbox),)

        odd_iou = torch.tensor(iou[0::2])
        even_iou = torch.tensor(iou[1::2])
        ones = torch.ones_like(odd_iou)
        zeros = torch.zeros_like(odd_iou)
        odd_iou_mask = torch.where(odd_iou >= even_iou, ones, zeros)
        even_iou_mask = torch.where(even_iou > odd_iou, ones, zeros)
        iou_mask = torch.empty_like(torch.tensor(iou))
        # 确保odd_iou_mask和even_iou_mask对应元素值的和为1
        assert (odd_iou_mask + even_iou_mask == ones).byte().all()

        iou_mask[0::2] = odd_iou_mask
        iou_mask[1::2] = even_iou_mask
    return iou_mask


def iou(bbox1, bbox2):
    assert bbox1.shape == bbox2.shape
    if not isinstance(bbox1, np.ndarray):
        bbox1 = bbox1.cpu().numpy()
        bbox2 = bbox2.cpu().numpy()
    tl = np.maximum(bbox1[:, 2:], bbox2[:, 2:])  # (len(gt_bbox),2)
    br = np.minimum(bbox1[:, :2], bbox2[:, :2])  # (len(gt_bbox),2)
    area_i = np.prod(br - tl, axis=1) * (tl < br).all(axis=1)  # (len(gt_bbox),)
    area_1 = np.prod(bbox1[:, :2] - bbox1[:, 2:], axis=1)  # (len(gt_bbox),)
    area_2 = np.prod(bbox2[:, :2] - bbox2[:, 2:], axis=1)  # (len(gt_bbox),)
    iou_out = area_i / (area_1 + area_2 - area_i)  # (len(gt_bbox),)
    return iou_out


if __name__ == '__main__':
    gt_bbox = np.array([[2,3,0,1]])
    pre_bbox = np.array([[3,4,1,2]])

    print(iou(gt_bbox, pre_bbox))



