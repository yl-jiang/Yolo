#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/8/17 上午10:25
# @Author  : jyl
# @File    : test.py
from V1.configs import opt
from V1.models.trainer import YoloV1Trainer
from V1.bbox_utils import xywh2yxyx
from V1.InferenceFunctions import inference
import torch
from PIL import Image
import os
import numpy as np
from V1.visualization import show_test
from torchvision import transforms


def test():
    trainer = YoloV1Trainer()
    model = trainer.yolo
    img_dict = precess_input_img(opt.test_img_path)
    vis_dict = validation(model, img_dict['resized_img'])
    vis_dict['GT_fname'] = img_dict['fname']
    vis_dict['GT_img'] = img_dict['ori_img']
    vis_dict['resized_img'] = np.transpose(img_dict['resized_img'], (0, 2, 3, 1))
    vis_dict['shape_scale'] = img_dict['shape_scale']
    show_test(vis_dict)


def precess_input_img(img_path):
    container = {}
    resized_img_list = []
    shape_scale_list = []
    fname_list = []
    ori_img = []
    test_img_fnames = os.listdir(img_path)
    torch_normailze = transforms.Compose([transforms.ToTensor(), transforms.Normalize(opt.mean, opt.std)])
    for img_fname in test_img_fnames:
        file_path = opt.test_img_path + '/' + str(img_fname)
        fname_list.append(img_fname)
        img = Image.open(file_path)
        shape_scale = np.array(img.size[::-1]) / np.array([opt.img_size, opt.img_size])
        shape_scale_list.append(shape_scale)
        if img.mode is not 'RGB':
            img = img.convert('RGB')
        ori_img.append(np.asarray(img))
        resized_img = img.resize(size=(opt.img_size, opt.img_size))
        resized_img = np.asarray(resized_img)
        resized_img = torch_normailze(resized_img)
        resized_img = resized_img.cpu().numpy()[None, ...]
        resized_img_list.append(resized_img)
    container['shape_scale'] = np.vstack(shape_scale_list)
    container['resized_img'] = np.vstack(resized_img_list)
    container['fname'] = fname_list
    container['ori_img'] = ori_img
    return container


def validation(net, imgs):
    assert imgs.shape[1:] == (3, opt.img_size, opt.img_size), f"Input image's size must be ({opt.img_size}, {opt.img_size}, 3)"
    net.eval()
    with torch.no_grad():
        img_num = len(imgs)
        imgs = torch.from_numpy(imgs).float().to(opt.device)
        pre = net(imgs).reshape(img_num, opt.S, opt.S, -1)
        class_pre = pre[:, :, :, opt.B*5:]  # [batch_size, 7, 7, 20]
        conf_pre = pre[:, :, :, :opt.B*5].reshape(img_num, opt.S, opt.S, -1, 5)  # [batch_size, 7, 7, 2, 5]
        conf_pre = conf_pre[:, :, :, :, -1].reshape(img_num, opt.S, opt.S, opt.B)  # [batch_size, 7, 7, 2]
        bbox_pre = pre[:, :, :, :opt.B*5].reshape(img_num, opt.S, opt.S, -1, 5)  # [batch_size, 7, 7, 2, 5]
        bbox_pre = bbox_pre[:, :, :, :, :4]  # [batch_size, 7, 7, 2, 4] / [x, y, w, h]
        bbox_pre = xywh2yxyx(bbox_pre, torch.ones(img_num, opt.S, opt.S, 5).to(opt.device))  # [ymax, xmax, ymin, xmin]
        batch_nms_bbox, batch_nms_conf, batch_nms_cls = inference(bbox_pre, conf_pre, class_pre)
        vis_dict = {'predict_bbox': batch_nms_bbox,
                    'predict_conf': batch_nms_conf,
                    'predict_class': batch_nms_cls}
        return vis_dict


if __name__ == '__main__':
    test()




