#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/8/3 下午10:13
# @Author  : jyl
# @File    : xywh2yxyx.py
import torch
from V1.configs import opt


def xywh2yxyx(bbox, target):
    """
    :param bbox:
        [[x,y,w,h], ...]
        shape:[2m, 4]
    :param target:
        shape:[batch, 7, 7, 30]
    :return:
        [[ymax,xmax,ymin,xmin], ...]
        shape:[2m, 4]
    """
    with torch.no_grad():
        bbox_shape = bbox.shape
        assert bbox.size(-1) == 4
        bbox = bbox.reshape(-1, 4)
        grid_id_target = target[:, :, :, 4].nonzero()[:, [1, 2]].float()  # [m, 2]
        grid_id_target = grid_id_target.repeat_interleave(repeats=opt.B, dim=0)  # [2*m, 2]
        assert bbox.size(0) == grid_id_target.size(0)
        center_x = (bbox[:, 0] + grid_id_target[:, 1]) * opt.grid_w
        center_y = (bbox[:, 1] + grid_id_target[:, 0]) * opt.grid_h
        half_w = torch.pow(bbox[:, 2], 2) * opt.img_size / 2
        half_h = torch.pow(bbox[:, 3], 2) * opt.img_size / 2
        x_max = center_x + half_w
        y_max = center_y + half_h
        x_min = center_x - half_w
        y_min = center_y - half_h

        bbox = torch.cat((y_max.reshape(-1, 1), x_max.reshape(-1, 1), y_min.reshape(-1, 1), x_min.reshape(-1, 1)), dim=1)
    return bbox.reshape(bbox_shape)


if __name__ == '__main__':
    bbox = torch.randn(5, 7, 7, 2, 4)
    xywh2yxyx(bbox)
