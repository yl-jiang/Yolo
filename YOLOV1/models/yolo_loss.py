#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/10 17:51
# @Author  : jyl
# @File    : yolo_loss.py
import torch
from torch.nn import functional
from V1.bbox_utils import bbox_iou, xywh2yxyx
from V1.configs import opt


class YoloLoss:
    def __init__(self, opt, predict, target):
        """
        :param opt:
            configs
        :param predict: outputs of yolo/tensor
            shape:[batch_size, 7, 7, 30]
        :param target:
            type:ndarray
            shape:[batch_size, 7, 7, 30]
        """
        self._check(predict, target)
        self.opt = opt
        self.row_elements = self.opt.B * 5 + self.opt.C
        self.predict = predict.float()
        self.target = target.float()

    @staticmethod
    def _check(*args):
        for i in args:
            if not isinstance(i, torch.Tensor):
                raise ValueError('All of input should be (torch.Tensor)')
            if i.size(-1) != (opt.B * 5 + opt.C):
                raise ValueError("The last dimension of YoloNet's output must be %d" % (opt.B * 5 + opt.C))

    def loss(self):
        # target:[batch_size, 7, 7, 30]
        coord_mask = self.target[:, :, :, [4]] > 0  # [batch, 7, 7, 1]
        noobj_mask = self.target[:, :, :, [4]] == 0  # [batch, 7, 7, 1]

        coord_mask = coord_mask.expand_as(self.target)  # [batch, 7, 7, 30]
        noobj_mask = noobj_mask.expand_as(self.target)  # [batch, 7, 7, 30]
        # [m, 30]
        coord_target = self.target[coord_mask].reshape(-1, self.row_elements)
        coord_predic = self.predict[coord_mask].reshape(-1, self.row_elements)
        # [m, 20]
        coord_class_target = coord_target[:, self.opt.B*5:]
        coord_class_predic = coord_predic[:, self.opt.B*5:]
        coord_bbox_confd_target = coord_target[:, :self.opt.B*5].reshape(-1, 5)  # [2*m, 5]
        coord_bbox_confd_predic = coord_predic[:, :self.opt.B*5].reshape(-1, 5)  # [2*m, 5]

        noobj_target = self.target[noobj_mask].reshape(-1, self.row_elements)  # [n, 30]
        noobj_predic = self.predict[noobj_mask].reshape(-1, self.row_elements)  # [n, 30]

        # 验证筛选出来的存在目标和不存在目标grid cell的个数之和等于总的grid cell数(m + n == batch*7*7)
        noobj_bbox_confd_target = noobj_target[:, :self.opt.B*5].reshape(-1, 5)  # [2*n, 5]
        noobj_bbox_confd_predic = noobj_predic[:, :self.opt.B*5].reshape(-1, 5)  # [2*n, 5]

        # 对不包含目标的grid cell只计算置信度损失，这些cell的置信度值预测值越小模型越好
        noobj_confd_target = noobj_bbox_confd_target[:, -1]  # [2*n, 1]
        noobj_confd_predic = noobj_bbox_confd_predic[:, -1]  # [2*n, 1]
        noobj_confd_loss = functional.mse_loss(noobj_confd_predic, noobj_confd_target, reduction='sum')

        # 对所有包含目标的cell计算置信度和分类误差
        coord_confd_loss = functional.mse_loss(coord_bbox_confd_target[:, -1], coord_bbox_confd_predic[:, -1],
                                               reduction='sum')
        coord_class_loss = functional.mse_loss(coord_class_predic, coord_class_target, reduction='sum')

        # 对所有包含目标的cell，选取与GT bbox iuo值最大的那个计算bbox误差
        gt_bbox = xywh2yxyx(coord_bbox_confd_target[:, :4], self.target)
        pre_bbox = xywh2yxyx(coord_bbox_confd_predic[:, :4], self.target)
        iou_mask = bbox_iou(gt_bbox, pre_bbox)  # (len(gt_bbox),)
        coord_pre = coord_bbox_confd_predic[iou_mask.byte()]
        coord_target = coord_bbox_confd_target[iou_mask.byte()]
        coord_xy_loss = functional.mse_loss(coord_pre[:, :2], coord_target[:, :2], reduction='sum')
        coord_hw_loss = functional.mse_loss(coord_pre[:, 2:4], coord_target[:, 2:4], reduction='sum')

        total_loss = opt.lambda_noobj * noobj_confd_loss + \
            coord_confd_loss + \
            opt.lambda_coord * coord_xy_loss + \
            opt.lambda_coord * coord_hw_loss + \
            coord_class_loss
        return total_loss / opt.batch_size


if __name__ == '__main__':
    yolo_loss = YoloLoss()
    yolo_loss.loss()