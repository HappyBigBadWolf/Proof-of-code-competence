
"""
Author:
    Yiqun Chen
Docs:
    Necessary modules for model.
"""

import os, sys
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from utils import utils


class DPDBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate, *args, **kwargs):
        super(DPDBottleneck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.drop_rate = drop_rate
        self._build()

    def _build(self):
        self.conv_1 = nn.Conv2d(self.in_channels, self.out_channels, 3, stride=1, padding=1)
        self.relu_1 = nn.ReLU()
        self.conv_2 = nn.Conv2d(self.out_channels, self.out_channels, 3, stride=1, padding=1)
        self.relu_2 = nn.ReLU()
        self.dropout = nn.Dropout(self.drop_rate)

    def forward(self, inp):
        out = self.relu_1(self.conv_1(inp))
        out = self.relu_2(self.conv_2(out))
        out = self.dropout(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, num_conv, in_channels, out_channels, *args, **kwargs):
        super(ResBlock, self).__init__()
        self.num_conv = num_conv
        self.in_channels = in_channels
        self.out_channels = out_channels
        self._build()

    def _build(self):
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, 3, stride=1, padding=1)
        layers = []
        for cnt in range(self.num_conv - 1):
            layers.extend([
                ("conv_"+str(cnt), nn.Conv2d(self.out_channels, self.out_channels, 3, stride=1, padding=1)), 
                ("relu_"+str(cnt), nn.ReLU())
            ])
        self.block = nn.Sequential(OrderedDict(layers))

    def forward(self, inp, *args, **kwargs):
        feat = self.conv(inp)
        feat = feat + self.block(feat)
        return feat