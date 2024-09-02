
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
from torch.nn.modules.utils import _pair
from collections import OrderedDict

from utils import utils

_BOTTLENECK = {}

def add_bottleneck(bottleneck):
    _BOTTLENECK[bottleneck.__name__] = bottleneck
    return bottleneck


class MultiHeadAttnModule(nn.Module):
    def __init__(self, in_channels, hid_channels, num_heads, *args, **kwargs):
        super(MultiHeadAttnModule, self).__init__()
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.num_heads = num_heads
        assert self.in_channels % self.num_heads == 0, "Value error"
        assert self.hid_channels % self.num_heads == 0, "Value error"
        self._build()

    def _build(self):
        self.linear_k = nn.Conv2d(self.in_channels, self.hid_channels, 1, 1, groups=self.num_heads)
        self.linear_q = nn.Conv2d(self.in_channels, self.hid_channels, 1, 1, groups=self.num_heads)
        self.linear_v = nn.Conv2d(self.in_channels, self.hid_channels, 1, 1, groups=self.num_heads)
        self.linear_o = nn.Conv2d(self.hid_channels, self.in_channels, 1, 1, groups=self.num_heads)
        self.multi_head_attn = nn.MultiheadAttention(self.hid_channels, self.num_heads)

    def forward(self, inp):
        feat_k = self.linear_k(inp)
        feat_q = self.linear_q(inp)
        feat_v = self.linear_v(inp)

        batch_size, channs, h, w = feat_k.shape

        feat_k = feat_k.reshape(batch_size, channs, h*w).permute(2, 0, 1).contiguous()
        feat_q = feat_q.reshape(batch_size, channs, h*w).permute(2, 0, 1).contiguous()
        feat_v = feat_v.reshape(batch_size, channs, h*w).permute(2, 0, 1).contiguous()

        out, _ = self.multi_head_attn(feat_q, feat_k, feat_v)
        out = out.permute(1, 2, 0).contiguous().reshape(batch_size, channs, h, w)
        out = self.linear_o(out)
        return out


class SplitAttnConv2d(nn.Module):
    """Split-Attention Conv2d
    """
    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, bias=True,
                 radix=2, reduction_factor=4, norm_layer=None,**kwargs):
        super(SplitAttnConv2d, self).__init__()
        padding = _pair(padding)
        inter_channels = max(in_channels*radix//reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.conv = nn.Conv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                            groups=groups*radix, bias=bias, **kwargs)
        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = norm_layer(channels*radix)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = norm_layer(inter_channels)
        self.fc2 = nn.Conv2d(inter_channels, channels*radix, 1, groups=self.cardinality)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        x = self.relu(x)

        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            if torch.__version__ < '1.5':
                splited = torch.split(x, int(rchannel//self.radix), dim=1)
            else:
                splited = torch.split(x, rchannel//self.radix, dim=1)
            gap = sum(splited) 
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)

        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)

        if self.radix > 1:
            if torch.__version__ < '1.5':
                attens = torch.split(atten, int(rchannel//self.radix), dim=1)
            else:
                attens = torch.split(atten, rchannel//self.radix, dim=1)
            out = sum([att*split for (att, split) in zip(attens, splited)])
        else:
            out = atten * x
        return out.contiguous()


class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class SplitAttnBottleneck(nn.Module):
    """ResNet Bottleneck
    """
    # pylint: disable=unused-argument
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 radix=2, cardinality=1, bottleneck_width=64,
                 avd=False, avd_first=False, dilation=1, is_first=False,
                 norm_layer=nn.BatchNorm2d, last_gamma=False):
        super(SplitAttnBottleneck, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(group_width)
        self.radix = radix
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        if radix >= 1:
            self.conv2 = SplitAttnConv2d(
                group_width, group_width, kernel_size=3,
                stride=stride, padding=dilation,
                dilation=dilation, groups=cardinality, bias=False,
                radix=radix, norm_layer=norm_layer)
        else:
            self.conv2 = nn.Conv2d(
                group_width, group_width, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation,
                groups=cardinality, bias=False)
            self.bn2 = norm_layer(group_width)

        self.conv3 = nn.Conv2d(
            group_width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes*4)

        if last_gamma:
            from torch.nn.init import zeros_
            zeros_(self.bn3.weight)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.avd and self.avd_first:
            out = self.avd_layer(out)

        out = self.conv2(out)
        if self.radix == 0:
            out = self.bn2(out)
            out = self.relu(out)

        if self.avd and not self.avd_first:
            out = self.avd_layer(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, *args, **kwargs):
        super(SEBlock, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self._build()

    def _build(self):
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))
        self.linear_1 = nn.Conv2d(self.in_channels, self.hidden_channels, 1, stride=1, padding=0)
        self.linear_2 = nn.Conv2d(self.hidden_channels, self.out_channels, 1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, org_feat, in_feat):
        feat = self.gmp(in_feat)
        feat = self.relu(self.linear_1(feat))
        feat = self.sigmoid(self.linear_2(feat))
        out_feat = in_feat * feat + org_feat
        return out_feat


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


@add_bottleneck
class DPDBottleneck(nn.Module):
    def __init__(self, cfg, in_channels=512, out_channels=1024, drop_rate=0.4, *args, **kwargs):
        super(DPDBottleneck, self).__init__()
        self.cfg = cfg
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


add_bottleneck(nn.Identity)


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



