#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/1 16:46
# @Author  : jyl
# @File    : train.py
from V1.configs import opt
from V1 import VocDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from V1.models.trainer import YoloV1Trainer
from V1.bbox_utils import xywh2yxyx
from V1.InferenceFunctions import inference
from V1.visualization import show_detection_result
import torch
import sys
from V1.data import choose_test_data


def train():
    dataset = VocDataset(is_train=True)
    dataset_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, drop_last=True)
    trainer = YoloV1Trainer()
    net = trainer.yolo
    patient = 0
    loss_tmp = float('inf')
    save_num = 0
    for epoch in range(trainer.before_epoch_num, opt.num_epoch):
        for i, (img, target) in tqdm(enumerate(dataset_loader)):
            img, target = img.to(opt.device), target.to(opt.device)
            loss = trainer.train_step(img, target, epoch)
            if i % opt.display_loss_every == 0:
                trainer.save(epoch, 'model_every')
                print('# Epoch {}  step {} training loss: {:.5f}'.format(epoch+1, i+1, loss))
                # 只保存最近的10个模型
                if save_num > 5:
                    save_num = 0
                if loss < loss_tmp:
                    trainer.save(epoch, 'model_best')
                    save_num += 1
                    loss_tmp = loss
                    patient = 0
                else:
                    if patient < 50:
                        patient += 1
                    else:
                        try:
                            sys.exit(0)
                        except SystemExit:
                            print(f'Early Stopping at epoch: {epoch} step: {i+1}')
        validation(net, epoch)


def validation(net, epoch):
    net.eval()
    with torch.no_grad():
        file_name, resize_img, img_input, GT_bboxes, GT_labels = choose_test_data(5)
        test_img_num = len(resize_img)
        img_input = torch.from_numpy(img_input).to(opt.device)  # img: [batch_size, 448, 448, 3]
        pre = net(img_input).reshape(img_input.shape[0], opt.S, opt.S, -1)
        class_pre = pre[:, :, :, opt.B*5:]  # [batch_size, 7, 7, 20]
        conf_pre = pre[:, :, :, :opt.B*5].reshape(test_img_num, opt.S, opt.S, -1, 5)  # [batch_size, 7, 7, 2, 5]
        conf_pre = conf_pre[:, :, :, :, -1].reshape(test_img_num, opt.S, opt.S, opt.B)  # [batch_size, 7, 7, 2]
        bbox_pre = pre[:, :, :, :opt.B*5].reshape(test_img_num, opt.S, opt.S, -1, 5)  # [batch_size, 7, 7, 2, 5]
        bbox_pre = bbox_pre[:, :, :, :, :4]  # [batch_size, 7, 7, 2, 4] / [x, y, w, h]
        bbox_pre = xywh2yxyx(bbox_pre, torch.ones(test_img_num, opt.S, opt.S, 5).to(opt.device))   # [batch_size, 7, 7, 2, 4] / [ymax, xmax, ymin, xmin]
        batch_nms_bbox, batch_nms_conf, batch_nms_cls = inference(bbox_pre, conf_pre, class_pre)
        visualization_dict = {'predict_bbox': batch_nms_bbox,
                              'predict_conf': batch_nms_conf,
                              'predict_class': batch_nms_cls,
                              'GT_img': resize_img,
                              'GT_bbox': GT_bboxes,
                              'GT_label': GT_labels,
                              'img_id': file_name,
                              'epoch': epoch}
        show_detection_result(visualization_dict)
    net.train()


if __name__ == '__main__':
    train()







