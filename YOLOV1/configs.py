#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/31 15:47
# @Author  : jyl
# @File    : configs.py
from pprint import pprint
import os
import torch
current_path = os.path.dirname(os.path.abspath(__file__))


class Config:
    current_path = current_path
    VOC_BBOX_LABEL_NAMES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                            'bus', 'car', 'cat', 'chair', 'cow',
                            'diningtable', 'dog', 'horse', 'motorbike', 'person',
                            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
    # data
    voc_data_dir = r'/home/dk/jyl/Object_Detection/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/'
    img_size = 448
    num_workers = 4
    test_num_workers = 4
    write_train = os.path.join(current_path, 'voc2012train.txt')
    write_test = os.path.join(current_path, 'voc2012test.txt')
    batch_size = 32
    transform_threshold = 0.2
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # model
    S = 7
    B = 2
    C = 20
    lambda_coord = 5
    lambda_noobj = 0.5
    grid_w = img_size / S
    grid_h = img_size / S

    # sigma for l1_smooth_loss
    rpn_sigma = 3.
    roi_sigma = 1.

    # optimizer
    momentum = 0.9
    weight_decay = 0.0005
    lr_decay = 0.1
    lr_decay_every = 30
    fe_lr = 0.01
    pre_lr = 0.01
    lr = 0.01

    # visualization
    env = 'faster-rcnn'  # visdom env
    port = 8097
    plot_every = 40  # vis every N iter
    display_loss_every = 100
    save_result_img_every = 100
    log_path = os.path.join(current_path, 'log', 'logging.log')
    log_config = os.path.join(current_path, 'log', 'logging.conf')
    test_img_path = os.path.join(current_path, 'test_imgs')
    img_save_path = os.path.join(current_path, 'visualization', 'result')

    # preset
    data = 'voc'
    pretrained_model = 'vgg16'

    # training
    num_epoch = 4000
    dropout_rate = 0.5

    # InferenceFunctions
    conf_threshold = 0.2
    iou_threshold = 0.5
    bbox_area_threshold = grid_w * grid_h

    use_adam = False  # Use Adam optimizer
    use_chainer = False  # try match everything as chainer
    use_drop = False  # use dropout in RoIHead
    # debug
    debug_file = '/tmp/debugf'
    test_num = 10000
    # model
    use_pretrain = False
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        device = 'cuda'
    else:
        device = 'cpu'
    save_path = os.path.join(current_path, 'checkpoints', 'model_best.pkl')
    ckpt_path = os.path.join(current_path, 'checkpoints')
    eval_every = 100

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)
        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() if not k.startswith('_')}


opt = Config()


