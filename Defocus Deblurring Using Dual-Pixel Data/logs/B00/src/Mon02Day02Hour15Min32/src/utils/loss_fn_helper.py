
"""
Author:
    Yiqun Chen
Docs:
    Help build loss functions.
"""

import os, sys
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

from utils import utils

_LOSS_FN = {}

def add_loss_fn(loss_fn):
    _LOSS_FN[loss_fn.__name__] = loss_fn
    return loss_fn

add_loss_fn(torch.nn.MSELoss)


@add_loss_fn
class SL2FL1Loss:
    def __init__(self, cfg, *args, **kwargs):
        super(SL2FL1Loss, self).__init__()
        self.cfg = cfg
        self.weights = self.cfg.LOSS_FN.WEIGHTS
        self._build()

    def _build(self):
        assert "L2SPAT" in self.weights.keys() and "L1FREQ" in self.weights.keys(), \
            "Weights of loss not found."

    def cal_loss(self, output, target):
        # Calculate loss in frequency domain. 
        fft_output = torch.fft.fft(torch.fft.fft(output, dim=2, norm="ortho"), dim=3, norm="ortho")
        fft_target = torch.fft.fft(torch.fft.fft(target, dim=2, norm="ortho"), dim=3, norm="ortho")
        assert fft_output.shape == output.shape, "ShapeError"
        assert fft_target.shape == target.shape, "ShapeError"
        assert output.shape == target.shape, "ShapeError"
        real_output = fft_output.real
        real_target = fft_target.real
        imag_output = fft_output.imag
        imag_target = fft_target.imag
        loss_real_l1 = F.l1_loss(real_output, real_target)
        loss_imag_l1 = F.l1_loss(imag_output, imag_target)

        loss_spatial_l2 = F.mse_loss(output, target)

        loss = self.weights.L1FREQ * (loss_real_l1 + loss_imag_l1) + self.weights.L2SPAT * loss_spatial_l2

        return loss


    def __call__(self, output, target):
        return self.cal_loss(output, target)


@add_loss_fn
class MSELoss:
    def __init__(self, cfg, *args, **kwargs):
        super(MSELoss, self).__init__()
        self.cfg = cfg
        self.args = args
        self.kwargs = kwargs
        self._build()

    def _build(self):
        self.loss_fn = nn.MSELoss()

    def cal_loss(self, output, target):
        loss = self.loss_fn(output, target)
        return loss

    def __call__(self, output, target):
        return self.cal_loss(output, target)


@add_loss_fn
class MAELoss:
    def __init__(self, cfg, *args, **kwargs):
        super(MAELoss, self).__init__()
        self.cfg = cfg
        self._build()

    def _build(self):
        self.loss_fn = nn.L1Loss()
    
    def cal_loss(self, output, target):
        loss = self.loss_fn(output, target)
        return loss

    def __call__(self, output, target):
        return self.cal_loss(output, target)
        


@add_loss_fn
class MyLossFn:
    def __init__(self, *args, **kwargs):
        super(MyLossFn, self).__init__()
        self._build()

    def _build(self):
        raise NotImplementedError("LossFn is not implemented yet.")

    def cal_loss(self, out, target):
        raise NotImplementedError("LossFn is not implemented yet.")

    def __call__(self, out, target):
        return self.cal_loss(out, target)


def build_loss_fn(cfg, *args, **kwargs):
    return _LOSS_FN[cfg.LOSS_FN.LOSS_FN](cfg, *args, **kwargs)


