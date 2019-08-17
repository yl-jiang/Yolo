#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/1 14:25
# @Author  : jyl
# @File    : darknet.py
import math
from torch import nn
from torch.nn import functional as F
from V1.configs import opt


class DarkNet(nn.Module):
    def __init__(self, pretrain=False):
        """
        input:
            (batch_size,3,448,448)
        output:
            (baych_size, 1000)
        ---------
        :param pretrain:
            whether use pretrained model
        """
        super(DarkNet, self).__init__()
        self.pretrain = pretrain

        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # (224, 224, 64)
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),  # (112, 112, 64)

            nn.Conv2d(64, 192, 3, padding=1),  # (112, 112, 192)
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),  # (56, 56, 192)

            nn.Conv2d(192, 128, 1),  # (56, 56, 128)
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),  # (56, 56, 256)
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, 1),  # (56, 56, 256)
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),  # (56, 56, 512)
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),  # (28, 28, 512)

            nn.Conv2d(512, 256, 1),  # (28, 28, 256)
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),  # (28, 28, 512)
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, 1),  # (28, 28, 256)
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),  # (28, 28, 512)
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, 1),  # (28, 28, 256)
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),  # (28, 28, 512)
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, 1),  # (28, 28, 256)
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),  # (28, 28, 512)
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 512, 1),  # (28, 28, 512)
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1),  # (28, 28, 1024)
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),  # (14, 14, 1024)

            nn.Conv2d(1024, 512, 1),  # (14, 14, 512)
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1),  # (14, 14, 1024)
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 512, 1),  # (14, 14, 512)
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1),  # (14, 14, 1024)
            nn.LeakyReLU(0.1, inplace=True)
        )
        if self.pretrain:
            self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        output = self.net(x)  # (batch,c=1024,h,w)
        if self.pretrain:
            output = F.avg_pool2d(output, kernel_size=(output.size(2), output.size(3)))  # (batch,c=1024,1,1)
            output.squeeze_()  # (batch,c=1024)
            print(output[0])
            print(output[1])
            output = F.softmax(self.fc(output))  # (batch, 1000)
        return output


class YoloV1(nn.Module):
    def __init__(self, img_size):
        super(YoloV1, self).__init__()
        self.feature = DarkNet()
        self.yolo = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, padding=1),  # (14, 14, 1024)
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, 3, stride=2, padding=1),  # (7, 7, 1024)
            nn.LeakyReLU(0.1),

            nn.Conv2d(1024, 1024, 3, padding=1),  # (7, 7, 1024)
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1),  # (7, 7, 1024)
            nn.LeakyReLU(0.1, inplace=True))

        self.flatten = Flatten()
        # (batch_size,7*7*1024) --> (batch_size, 4096)
        self.fc1 = nn.Linear((math.ceil(img_size / 64) * math.ceil(img_size / 64) * 1024), 4096)
        self.dropout = nn.Dropout(p=opt.dropout_rate)
        # (batch_size, 4096) --> (batch_size, 7*7*30)
        self.fc2 = nn.Linear(4096, opt.S*opt.S*(opt.B*5+opt.C))

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        feature = self.feature(x)
        output = self.yolo(feature)
        output = self.flatten(output)
        output = F.leaky_relu(self.dropout(self.fc1(output)), 0.1)
        output = self.fc2(output)
        # output: [batch_size, 7*7*30]
        return output


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


if __name__ == '__main__':
    import torch
    import numpy as np
    rand_imgs = torch.as_tensor(np.random.randint(low=0, high=3010560,size=[5,3,448,448])).float()
    net = YoloV1(448)
    pre = net(rand_imgs)
    a = pre[0]
    b = pre[1]
    d = a == b
    c = 1
