
"""
Author:
    Yiqun Chen
Docs:
    Decoder classes.
"""

import os, sys
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
from collections import OrderedDict
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

from utils import utils
from .modules import *

_DECODER = {}

def add_decoder(decoder):
    _DECODER[decoder.__name__] = decoder
    return decoder

@add_decoder
class DPDDecoder(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super(DPDDecoder, self).__init__()
        self.cfg = cfg
        self._build()

    def _build(self):
        self.block_1_1, self.block_1_2 = self._build_block(2, 1024, 512)
        self.block_2_1, self.block_2_2 = self._build_block(2, 512, 256)
        self.block_3_1, self.block_3_2 = self._build_block(2, 256, 128)
        self.block_4_1, self.block_4_2 = self._build_block(2, 128, 64)
        self.out_block = nn.Sequential(
            nn.Conv2d(64, 3, 3, stride=1, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(3, 3, 1, stride=1), 
            nn.Sigmoid(), 
        )

    def _build_block(self, num_conv, in_channels, out_channels):
        block_1 = nn.Sequential(OrderedDict([
            ("upsampling", nn.UpsamplingNearest2d(scale_factor=2)), 
            ("conv", nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1))
        ]))
        layer_list = []
        for idx in range(num_conv):
            layer_list.append(
                ("conv_"+str(idx), nn.Conv2d(out_channels*2 if idx == 0 else out_channels, out_channels, 3, stride=1, padding=1))
            )
            layer_list.append(
                ("relu_"+str(idx), nn.ReLU())
            )
        block_2 = nn.Sequential(OrderedDict(layer_list))
        return block_1, block_2

    def forward(self, inp, *args, **kwargs):
        enc_1, enc_2, enc_3, enc_4, bottleneck = inp
        
        # decoder block 1
        out = self.block_1_1(bottleneck)
        out = torch.cat([out, enc_4], dim=1)
        out = self.block_1_2(out)

        # decoder block 2
        out = self.block_2_1(out)
        out = torch.cat([out, enc_3], dim=1)
        out = self.block_2_2(out)

        # decoder block 3
        out = self.block_3_1(out)
        out = torch.cat([out, enc_2], dim=1)
        out = self.block_3_2(out)

        # decoder block 4
        out = self.block_4_1(out)
        out = torch.cat([out, enc_1], dim=1)
        out = self.block_4_2(out)
        
        out = self.out_block(out)
        return out


@add_decoder
class DPDDecoderV2(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super(DPDDecoderV2, self).__init__()
        self.cfg = cfg
        self._build()

    def _build(self):
        self.block_1_1, self.block_1_2 = self._build_block(2, 1024, 512)
        self.block_2_1, self.block_2_2 = self._build_block(2, 512, 256)
        self.block_3_1, self.block_3_2 = self._build_block(2, 256, 128)
        self.block_4_1, self.block_4_2 = self._build_block(2, 128, 64)
        self.out_block = nn.Sequential(
            nn.Conv2d(64, 3, 3, stride=1, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(3, 3, 1, stride=1), 
            nn.Sigmoid(), 
        )

    def _build_block(self, num_conv, in_channels, out_channels):
        block_1 = nn.Sequential(OrderedDict([
            ("upsampling", nn.UpsamplingNearest2d(scale_factor=2)), 
            ("conv", nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1))
        ]))
        layer_list = []
        for idx in range(num_conv):
            layer_list.append(
                ("conv_"+str(idx), nn.Conv2d(out_channels*2 if idx == 0 else out_channels, out_channels, 3, stride=1, padding=1))
            )
            layer_list.append(
                ("relu_"+str(idx), nn.ReLU())
            )
        block_2 = nn.Sequential(OrderedDict(layer_list))
        return block_1, block_2

    def forward(self, inp: list, *args, **kwargs):
        enc_1, enc_2, enc_3, enc_4, bottleneck = inp
        
        # decoder block 1
        out = self.block_1_1(bottleneck)
        out = torch.cat([out, enc_4], dim=1)
        dec_1 = self.block_1_2(out)

        # decoder block 2
        out = self.block_2_1(dec_1)
        out = torch.cat([out, enc_3], dim=1)
        dec_2 = self.block_2_2(out)

        # decoder block 3
        out = self.block_3_1(dec_2)
        out = torch.cat([out, enc_2], dim=1)
        dec_3 = self.block_3_2(out)

        # decoder block 4
        out = self.block_4_1(dec_3)
        out = torch.cat([out, enc_1], dim=1)
        dec_4 = self.block_4_2(out)
        
        out = self.out_block(dec_4)

        feats = inp
        feats += (dec_1, dec_2, dec_3, dec_4)
        return out, feats


@add_decoder
class DPDDecoderV3(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super(DPDDecoderV3, self).__init__()
        self.cfg = cfg
        self._build()

    def _build(self):
        self.block_1 = self._build_block(3, 1024, 512)
        self.block_2 = self._build_block(3, 512, 256)
        self.block_3 = self._build_block(3, 256, 128)
        self.block_4 = self._build_block(3, 128, 64)
        self.out_block = nn.Sequential(
            nn.Conv2d(64, 3, 3, stride=1, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(3, 3, 1, stride=1), 
            nn.Sigmoid(), 
        )

    def _build_block(self, num_conv, in_channels, out_channels):
        block = nn.Sequential(OrderedDict([
            ("upsampling", nn.UpsamplingNearest2d(scale_factor=2)), 
            ("resblock", ResBlock(num_conv=num_conv, in_channels=in_channels, out_channels=out_channels))
        ]))
        return block

    def forward(self, inp, *args, **kwargs):
        bottleneck = inp
        
        # decoder block 1
        dec_1 = self.block_1(bottleneck)

        # decoder block 2
        dec_2 = self.block_2(dec_1)
        
        # decoder block 3
        dec_3 = self.block_3(dec_2)

        # decoder block 4
        dec_4 = self.block_4(dec_3)
        
        out = self.out_block(dec_4)

        feats = (inp, dec_1, dec_2, dec_3, dec_4)
        return out, feats


@add_decoder
class DPDDecoderV4(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super(DPDDecoderV4, self).__init__()
        self.cfg = cfg
        self._build()

    def _build(self):
        self.block_1_1, self.block_1_2 = self._build_block(2, 1024, 512)
        self.block_2_1, self.block_2_2 = self._build_block(2, 512, 256)
        self.block_3_1, self.block_3_2 = self._build_block(2, 256, 128)
        self.block_4_1, self.block_4_2 = self._build_block(2, 128, 64)
        self.attn = nn.Sequential(*[AttnBlock2D(1024, 1024 // 4) for i in range(4)])
        self.out_block = nn.Sequential(
            nn.Conv2d(64, 3, 3, stride=1, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(3, 3, 1, stride=1), 
            nn.Sigmoid(), 
        )

    def _build_block(self, num_conv, in_channels, out_channels):
        block_1 = nn.Sequential(OrderedDict([
            ("upsampling", nn.UpsamplingNearest2d(scale_factor=2)), 
            ("conv", nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1))
        ]))
        layer_list = []
        for idx in range(num_conv):
            layer_list.append(
                ("conv_"+str(idx), nn.Conv2d(out_channels*2 if idx == 0 else out_channels, out_channels, 3, stride=1, padding=1))
            )
            layer_list.append(
                ("relu_"+str(idx), nn.ReLU())
            )
        block_2 = nn.Sequential(OrderedDict(layer_list))
        return block_1, block_2

    def forward(self, inp, *args, **kwargs):
        enc_1, enc_2, enc_3, enc_4, bottleneck = inp

        out = self.attn(bottleneck)
        
        # decoder block 1
        out = self.block_1_1(out)
        out = torch.cat([out, enc_4], dim=1)
        out = self.block_1_2(out)

        # decoder block 2
        out = self.block_2_1(out)
        out = torch.cat([out, enc_3], dim=1)
        out = self.block_2_2(out)

        # decoder block 3
        out = self.block_3_1(out)
        out = torch.cat([out, enc_2], dim=1)
        out = self.block_3_2(out)

        # decoder block 4
        out = self.block_4_1(out)
        out = torch.cat([out, enc_1], dim=1)
        out = self.block_4_2(out)
        
        out = self.out_block(out)
        return out


@add_decoder
class UNetDecoder(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super(UNetDecoder, self).__init__()
        self.cfg = cfg
        self._build()

    def _build(self):
        model = torch.hub.load(
            'mateuszbuda/brain-segmentation-pytorch', 'unet', 
            in_channels=3, out_channels=1, init_features=32, pretrained=True
        )
        self.model = nn.ModuleDict({
            "decoder4": model.decoder4, 
            "decoder3": model.decoder3,
            "decoder2": model.decoder2,
            "decoder1": model.decoder1,
            "upconv4": model.upconv4, 
            "upconv3": model.upconv3,
            "upconv2": model.upconv2,
            "upconv1": model.upconv1,   
            "conv": model.conv, 
        })
        
    def forward(self, data, *args, **kwargs):
        enc1, enc2, enc3, enc4, bottleneck = data
        dec4 = self.model["upconv4"](bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.model["decoder4"](dec4)
        dec3 = self.model["upconv3"](dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.model["decoder3"](dec3)
        dec2 = self.model["upconv2"](dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.model["decoder2"](dec2)
        dec1 = self.model["upconv1"](dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.model["decoder1"](dec1)
        raise NotImplementedError("Method UNetDecoder.forward is not implemented.")



if __name__ == "__main__":
    print(_DECODER)
    model = _DECODER["UNetDecoder"](None)
    print(_DECODER)