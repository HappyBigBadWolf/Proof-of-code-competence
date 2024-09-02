
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


class FeatAttnDecBlock(nn.Module):
    def __init__(self, num_blocks, channels, *args, **kwargs):
        super(FeatAttnDecBlock, self).__init__()
        self.channels = channels
        self.num_blocks = num_blocks
        self._build()

    def _build(self):
        layers = [FeatAttnBlock(self.channels) for i in range(self.num_blocks)]
        self.model = nn.Sequential(*layers)

    def forward(self, inp):
        out = self.model(inp)
        return out


class FeatAttnBlock(nn.Module):
    def __init__(self, channels, *args, **kwargs):
        super(FeatAttnBlock, self).__init__()
        self.channels = channels
        self._build()

    def _build(self):
        self.conv_1 = nn.Conv2d(self.channels, self.channels, 3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv_2 = nn.Conv2d(self.channels, self.channels, 3, stride=1, padding=1)
        self.pixel_attn = PixelAttn(self.channels)
        self.chann_attn = ChannAttn(self.channels)

    def forward(self, inp):
        feat = self.relu(self.conv_1(inp))
        feat = feat + inp
        feat = self.conv_2(feat)
        feat = self.chann_attn(feat)
        feat = self.pixel_attn(feat)
        feat = feat + inp
        return feat


class AttnBlock2D(nn.Module):
    def __init__(self, in_channels, hid_channels, *args, **kwargs):
        super(AttnBlock2D, self).__init__()
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self._build()

    def _build(self):
        self.lin_k = nn.Conv2d(self.in_channels, self.hid_channels, 1)
        self.lin_q = nn.Conv2d(self.in_channels, self.hid_channels, 1)
        self.lin_v = nn.Conv2d(self.in_channels, self.hid_channels, 1)
        self.lin_o = nn.Conv2d(self.hid_channels, self.in_channels, 1)

    def forward(self, inp):
        key = self.lin_k(inp)
        query = self.lin_q(inp)
        value = self.lin_v(inp)
        batch, channels, height, width = key.shape
        key = key.reshape(batch, channels, -1)
        query = query.reshape(batch, channels, -1)
        value = value.reshape(batch, channels, -1)
        weight = F.softmax(torch.bmm(key.transpose(1, 2), query) / torch.sqrt(torch.tensor(channels, dtype=torch.float)), dim=2)
        out = torch.bmm(weight, value.transpose(1, 2))
        assert out.shape[2] == channels, "ShapeError"
        out = out.transpose(1, 2).reshape(batch, channels, height, width)
        out = self.lin_o(out)
        return out
        

class PixelAttn(nn.Module):
    def __init__(self, channels, *args, **kwargs):
        super(PixelAttn, self).__init__()
        self.channels = channels
        self._build()

    def _build(self):
        self.conv_1 = nn.Conv2d(self.channels, self.channels, 1)
        self.conv_2 = nn.Conv2d(self.channels, 1, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp):
        weight = self.relu(self.conv_1(inp))
        weight = self.sigmoid(self.conv_2(weight))
        out = inp * weight
        return out


class ChannAttn(nn.Module):
    def __init__(self, channels, *args, **kwargs):
        super(ChannAttn, self).__init__()
        self.channels = channels
        self._build()

    def _build(self):
        self.conv_1 = nn.Conv2d(self.channels, self.channels, 1)
        self.conv_2 = nn.Conv2d(self.channels, self.channels, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp):
        weight = inp.mean(dim=[2, 3], keepdim=True)
        weight = self.relu(self.conv_1(weight))
        weight = self.sigmoid(self.conv_2(weight))
        out = inp * weight
        return out


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