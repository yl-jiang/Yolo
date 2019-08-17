#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/13 14:20
# @Author  : jyl
# @File    : trainer.py
from V1.configs import opt
from V1 import VocDataset
from torch.utils.data import DataLoader
from V1 import YoloV1
from torch import nn
import os
from V1.models.yolo_loss import YoloLoss
from torchnet.meter import AverageValueMeter
import torch


class YoloV1Trainer(nn.Module):
    def __init__(self):
        super(YoloV1Trainer, self).__init__()
        if opt.gpu_available:
            device = 'cuda'
        else:
            device = 'cpu'
        # self.yolo: [batch_size, 7*7*30]
        self.yolo = YoloV1(opt.img_size).to(device)
        self.optimizer = self.get_optimizer()
        self.before_epoch_num = self.use_pretrain()

        self.dataset = VocDataset(is_train=True)
        self.testset = VocDataset(is_train=False)
        self.dataset_loader = DataLoader(self.dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
        self.testset_loader = DataLoader(self.testset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
        self.loss_meter = AverageValueMeter()

    def use_pretrain(self, use_pretain=True, load_optimizer=True):
        if use_pretain and os.path.exists(opt.save_path):
            state_dict = torch.load(opt.save_path)
            before_epoch_num = state_dict['epoch']
            if 'model' in state_dict.keys():
                self.yolo.load_state_dict(state_dict['model'])
                if load_optimizer and 'optimizer' in state_dict.keys():
                    print('Loading Pre-training Model ... ...!')
                    self.optimizer.load_state_dict(state_dict['optimizer'])
        else:
            before_epoch_num = 0
            print('Faild To Use Pre-training Mdel! Strat training ... ...!')

        return before_epoch_num

    def save(self, epoch, ckpt_name):
        model_state = self.yolo.state_dict()
        optimizer_state = self.optimizer.state_dict()
        state_dict = dict((('model', model_state),
                           ('optimizer', optimizer_state),
                           ('epoch', epoch),
                           ('loss', self.loss_meter.mean)))
        if not os.path.exists(opt.ckpt_path):
            os.makedirs(opt.ckpt_path)
        torch.save(state_dict, opt.ckpt_path + '/' + ckpt_name + '.pkl')

    def train_step(self, img, target, epoch):
        loss = self.forward(img, target)
        self.adjust_learning_rate(epoch + self.before_epoch_num)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss_meter.add(loss.cpu().item())
        return self.loss_meter.mean

    def forward(self, img, target):
        pred = self.yolo(img)
        pred = pred.reshape(-1, opt.S, opt.S, opt.B*5+opt.C)
        loss = YoloLoss(opt=opt, predict=pred, target=target).loss()
        return loss

    def get_optimizer(self):
        params = list()
        params.append({'params': self.yolo.feature.parameters(), 'lr': opt.fe_lr})
        params.append({'params': self.yolo.yolo.parameters(), 'lr': opt.pre_lr})
        return torch.optim.SGD(params, lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        if epoch < 150:
            pass
        elif 150 <= epoch < 450:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 0.005
        elif 450 <= epoch < 600:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 0.0025
        else:
            for param_group in self.optimizer.param_groups:
                lr = param_group['lr']
                param_group['lr'] = 0.0001
