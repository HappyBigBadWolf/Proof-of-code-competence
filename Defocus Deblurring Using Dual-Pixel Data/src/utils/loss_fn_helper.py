
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
class SSIMLoss:
    def __init__(self, cfg, *args, **kwargs):
        super(SSIMLoss, self).__init__()
        self.cfg = cfg
        self._build()

    def _build(self):
        self.device = torch.device("cpu" if len(self.cfg.GENERAL.GPU) == 0 else "cuda:"+str(self.cfg.GENERAL.GPU[0]))

    def cal_loss(self, output, target):
        loss = 1 - utils.cal_ssim_pt(output, target, data_range=1.0, multichannel=True, device=self.device)
        return loss

    def __call__(self, output, target):
        return self.cal_loss(output, target)


@add_loss_fn
class ColorLoss:
    def __init__(self, cfg, eps=1e-6, *args, **kwargs):
        super(ColorLoss, self).__init__()
        self.cfg = cfg
        self.eps = eps
        self._build()

    def _build(self):
        self.cosine = nn.CosineSimilarity(dim=1)
        self.loss_fn = nn.CosineEmbeddingLoss()
        self.l1_loss_fn = nn.L1Loss()

    def cal_loss(self, output, target):
        if output.shape != target.shape:
            raise ValueError("Expect output and target have same shape, but got {} and {}".format(output.shape, target.shape))
        if output.max() > 1.0 or output.min() < 0.0 or target.max() > 1.0 or target.min() < 0.0:
            raise ValueError("Input should in range [0, 1]")
        batch_size, _, height, width = output.shape
        ones = torch.ones(batch_size, device=output.device)
        output_hsv = utils.rgb_to_hsv(output)
        target_hsv = utils.rgb_to_hsv(target)

        # output_hsv = output_hsv.reshape(batch_size, 3, -1)
        # target_hsv = target_hsv.reshape(batch_size, 3, -1)

        '''loss_h = self.loss_fn(output_hsv[:, 0, :], target_hsv[:, 0, :], ones)
        loss_s = self.loss_fn(output_hsv[:, 1, :], target_hsv[:, 1, :], ones)
        loss_v = self.loss_fn(output_hsv[:, 2, :], target_hsv[:, 2, :], ones)'''
        '''loss_h = self.l1_loss_fn(output_hsv[:, 0, :], target_hsv[:, 0, :])
        loss_s = self.l1_loss_fn(output_hsv[:, 1, :], target_hsv[:, 1, :])
        loss_v = self.l1_loss_fn(output_hsv[:, 2, :], target_hsv[:, 2, :])'''

        # loss = loss_h + loss_s + loss_v
        loss = self.l1_loss_fn(output_hsv, target_hsv)
        return loss


    def __call__(self, output, target):
        return self.cal_loss(output, target)


@add_loss_fn
class MSESSIMLoss:
    def __init__(self, cfg, *args, **kwargs):
        super(MSESSIMLoss, self).__init__()
        self.cfg = cfg
        self.weights = self.cfg.LOSS_FN.WEIGHTS
        self._build()

    def _build(self):
        self.loss_fn_mse = MSELoss(self.cfg)
        self.loss_fn_ssim = SSIMLoss(self.cfg)
        assert "L2SPAT" in self.weights.keys() and "SSIM" in self.weights.keys(), \
            "Weights of loss are not found"

    def cal_loss(self, output, target):
        loss_mse = self.loss_fn_mse(output, target)
        loss_ssim = self.loss_fn_ssim(output, target)
        loss = self.weights.L2SPAT * loss_mse + self.weights.SSIM * loss_ssim
        return loss

    def __call__(self, output, target):
        return self.cal_loss(output, target)


@add_loss_fn
class MSEColorSSIMLoss:
    def __init__(self, cfg, *args, **kwargs):
        super(MSEColorSSIMLoss, self).__init__()
        self.cfg = cfg
        self.weights = self.cfg.LOSS_FN.WEIGHTS
        self._build()

    def _build(self):
        self.loss_fn_mse = MSELoss(self.cfg)
        self.loss_fn_ssim = SSIMLoss(self.cfg)
        self.loss_fn_color = ColorLoss(self.cfg)
        assert "L2SPAT" in self.weights.keys() and "SSIM" in self.weights.keys() and "COLOR" in self.weights.keys(), \
            "Weights of loss are not found"

    def cal_loss(self, output, target):
        loss_mse = self.loss_fn_mse(output, target)
        loss_ssim = self.loss_fn_ssim(output, target)
        loss_color = self.loss_fn_color(output, target)
        loss = self.weights.L2SPAT * loss_mse + self.weights.SSIM * loss_ssim + self.weights.COLOR * loss_color
        return loss

    def __call__(self, output, target):
        return self.cal_loss(output, target)


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


