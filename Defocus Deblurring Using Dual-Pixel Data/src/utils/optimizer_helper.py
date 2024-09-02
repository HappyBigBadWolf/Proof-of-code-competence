
"""
Author  Yiqun Chen
Docs    Help build optimizer.
"""

import os, sys
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

from utils import utils

_OPTIMIZER = {}

def add_optimizer(optim_func):
    _OPTIMIZER[optim_func.__name__] = optim_func
    return optim_func

@add_optimizer
def SGD(cfg, model):
    lr = cfg.OPTIMIZER.LR
    momentum = cfg.OPTIMIZER.MOMENTUM if hasattr(cfg.OPTIMIZER, "MOMENTUM") else 0
    dampening = cfg.OPTIMIZER.DAMPENING if hasattr(cfg.OPTIMIZER, "DAMPENING") else 0
    weight_decay = cfg.OPTIMIZER.WEIGHT_DECAY if hasattr(cfg.OPTIMIZER, "WEIGHT_DECAY") else 0
    nesterov = cfg.OPTIMIZER.NESTEROV if hasattr(cfg.OPTIMIZER, "NESTEROV") else False
    optimizer = torch.optim.SGD(
        [
            {"params": model.encoder.parameters(), "lr": lr}, 
            {"params": model.decoder.parameters(), "lr": lr}, 
            {"params": model.backbone.parameters(), "lr": lr}, 
        ], 
        lr=lr, 
        momentum=momentum, 
        dampening=dampening, 
        weight_decay=weight_decay, 
        nesterov=nesterov, 
    )
    raise NotImplementedError("Optimizer SGD is not implemented yet.")
    return optimizer

@add_optimizer
def Adam(cfg, model):
    lr = cfg.OPTIMIZER.LR
    weight_decay = cfg.OPTIMIZER.WEIGHT_DECAY if hasattr(cfg.OPTIMIZER, "WEIGHT_DECAY") else 0
    betas = cfg.OPTIMIZER.BETAS if hasattr(cfg.OPTIMIZER, "BETAS") else (0.9, 0.999)
    eps = cfg.OPTIMIZER.EPS if hasattr(cfg.OPTIMIZER, "EPS") else 1E-8
    amsgrad = cfg.OPTIMIZER.AMSGRAD if hasattr(cfg.OPTIMIZER, "AMSGRAD") else False
    optimizer = torch.optim.Adam(
        [
            {"params": model.encoder.parameters(), "lr": lr}, 
            {"params": model.decoder.parameters(), "lr": lr}, 
            {"params": model.bottleneck.parameters(), "lr": lr}, 
        ], 
        lr=lr, 
        betas=betas, 
        eps=eps, 
        weight_decay=weight_decay, 
        amsgrad=amsgrad, 
    )
    return optimizer

@add_optimizer
def AdamW(cfg, model):
    lr = cfg.OPTIMIZER.LR
    weight_decay = cfg.OPTIMIZER.WEIGHT_DECAY if hasattr(cfg.OPTIMIZER, "WEIGHT_DECAY") else 0.01
    betas = cfg.OPTIMIZER.BETAS if hasattr(cfg.OPTIMIZER, "BETAS") else (0.9, 0.999)
    eps = cfg.OPTIMIZER.EPS if hasattr(cfg.OPTIMIZER, "EPS") else 1E-8
    amsgrad = cfg.OPTIMIZER.AMSGRAD if hasattr(cfg.OPTIMIZER, "AMSGRAD") else False
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": lr}, 
            {"params": model.decoder.parameters(), "lr": lr}, 
            {"params": model.bottleneck.parameters(), "lr": lr}, 
        ], 
        lr=lr, 
        betas=betas, 
        eps=eps, 
        weight_decay=weight_decay, 
        amsgrad=amsgrad, 
    )
    return optimizer

def build_optimizer(cfg, model, *args, **kwargs):
    optimizer = _OPTIMIZER[cfg.OPTIMIZER.OPTIMIZER](cfg, model)
    # raise NotImplementedError("Function build_optimizer is not implemented.")
    return optimizer