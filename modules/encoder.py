# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from torch import cat, Tensor
from torch.nn import Module
import torch.nn as nn
import numpy as np

__author__ = 'Wu Qianyang, Tao Shengqi, Yang Xingyu'
__docformat__ = 'reStructuredText'
__all__ = ['Eecoder']

cfg = {
    'TWY8':[16,'M',16,'M',32,'M',32,'M',32,'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}


class Encoder(Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int)\
            ->None:
        super().__init__()
        self.cnn_name = 'TWY8'
        self.features = self._make_layers(cfg[self.cnn_name])

        # view
        self.fc1 = nn.Linear(32 * 81 * 4 ,4096) # First dimension depends on the size of input data
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, out_dim)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)

        out = F.log_softmax(out, dim=1)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 1
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2,)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
