
"""
Author:
    Yiqun Chen
Docs:
    Utilities, should not call other custom modules.
"""

import os, sys, copy, functools, time, contextlib
import torch, torchvision
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2

@contextlib.contextmanager
def log_info(msg="", level="INFO", state=False, logger=None):
    log = print if logger is None else logger.log_info
    _state = "[{:<8}]".format("RUNNING") if state else ""
    log("[{:<20}] [{:<8}] {} {}".format(time.asctime(), level, _state, msg))
    yield
    if state:
        _state = "[{:<8}]".format("DONE") if state else ""
        log("[{:<20}] [{:<8}] {} {}".format(time.asctime(), level, _state, msg))

def log_info_wrapper(msg, logger=None):
    """
    Decorate factory.
    """
    def func_wraper(func):
        """
        The true decorate.
        """
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            # log = print if logger is None else logger.log_info
            # log("[{:<20}] [{:<8}]".format(time.asctime(), "RUNNING"), msg)
            with log_info(msg=msg, level="INFO", state=True, logger=logger):
                res = func(*args, **kwargs)
            # log("[{:<20}] [{:<8}]".format(time.asctime(), "DONE"), msg)
            return res
        return wrapped_func
    return func_wraper

def inference(model, data, device):
    """
    Info:
        Inference once, without calculate any loss.
    Args:
        - model (nn.Module):
        - data (dict): necessary keys: "l_view", "r_view"
        - device (torch.device)
    Returns:
        - out (Tensor): predicted.
    """
    def _inference(model, data, device):
        l_view, r_view = data["l_view"], data["r_view"]
        assert len(l_view.shape) == len(r_view.shape) == 4, "Incorrect shape."
        inp = torch.cat([l_view, r_view], dim=1)
        inp = inp.to(device)
        out = model(inp)
        return out, 

    def _inference_V2(model, data, device):
        l_view, r_view = data["l_view"], data["r_view"]
        assert len(l_view.shape) == len(r_view.shape) == 4, "Incorrect shape."
        inp = torch.cat([l_view, r_view], dim=1)
        inp = inp.to(device)
        out, feats = model(inp)
        return out, feats

    return _inference(model, data, device) 

def inference_and_cal_loss(model, data, loss_fn, device):
    """
    Info:
        Execute inference and calculate loss, sychronize the train and evaluate progress. 
    Args:
        - model (nn.Module):
        - data (dict): necessary keys: "l_view", "r_view"
        - loss_fn (callable): function or callable instance.
        - device (torch.device)
    Returns:
        - out (Tensor): predicted.
        - loss (Tensor): calculated loss.
    """
    def _infer_and_cal_loss(model, data, loss_fn, device):
        # NOTE 2021-01-22
        out, *_ = inference(model, data, device)
        target = data["target"].to(device)
        loss = loss_fn(out, target)
        return out, loss
    
    def _infer_and_cal_loss_V2(model, data, loss_fn, device):
        # NOTE 2021-01-22
        l_view, r_view, target = data["l_view"].to(device), data["r_view"].to(device), data["target"].to(device)
        assert len(l_view.shape) == len(r_view.shape) == 4, "Incorrect shape."
        batch_size = l_view.shape[0]

        inp_1 = torch.cat([l_view, r_view], dim=1)
        inp_2 = torch.cat([torch.clone(target), torch.clone(target)], dim=1)

        out, feats = model(torch.cat([inp_1, inp_2], dim=0))
        out_1, out_2 = out[0: batch_size], out[batch_size: ]
        feats_1, feats_2 = [feat[0: batch_size] for feat in feats], [feat[batch_size: ] for feat in feats]
        assert len(feats_1) == len(feats_2) == len(feats), "Number of features mismatch."

        loss_1 = loss_fn(out_1, target)
        loss_2 = loss_fn(out_2, target)
        loss_sum = loss_fn(out_1, out_2)
        loss = loss_1 + loss_2 + loss_sum
        for feat_1, feat_2 in zip(feats_1, feats_2):
            loss = loss + loss_fn(feat_1, feat_2)
        return out_1, loss

    # return _infer_and_cal_loss(model, data, loss_fn, device)
    return _infer_and_cal_loss(model, data, loss_fn, device)

def cal_and_record_metrics(phase, epoch, output, target, metrics_logger, logger=None):
    output = output.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    batch_size = output.shape[0]
    for idx in range(batch_size):
        metrics_logger.cal_metrics(phase, epoch, target[idx], output[idx], data_range=1)

def save_image(output, mean, norm, path2file):
    """
    Info:
        Save output to specific path.
    Args:
        - output (Tensor | ndarray): takes value from range [0, 1].
        - mean (float):
        - norm (float): 
        - path2file (str | os.PathLike):
    Returns:
        - (bool): indicate succeed or not.
    """
    if isinstance(output, torch.Tensor):
        output = output.numpy()
    output = ((output.transpose((1, 2, 0)) * norm) + mean).astype(np.uint16)
    try:
        cv2.imwrite(path2file, output)
        return True
    except:
        return False

def resize(img: torch.Tensor, size: list or tuple, logger=None):
    """
    Info:
        Resize the input image. 
    Args:
        - img (torch.Tensor):
        - size (tuple | int): target size of image.
        - logger (Logger): record running information, if None, direct message to terminal.
    Returns:
        - img (torch.Tensor): image with target size. 
    """
    org_shape = img.shape
    if len(org_shape) == 2:
        img = img.unsqueeze(0).unsqueeze(0)
    elif len(org_shape) == 3:
        img = img.unsqueeze(0)
    elif len(org_shape) == 4:
        pass
    else:
        raise NotImplementedError("Function to deal with image with shape {} is not implememted yet.".format(org_shape))
    img = F.interpolate(img, size=size, mode="bilinear", align_corners=False)
    img = img.reshape(org_shape)
    return img

def set_device(model: torch.nn.Module, gpu_list: list, logger=None):
    with log_info(msg="Set device for model.", level="INFO", state=True, logger=logger):
        if not torch.cuda.is_available():
            with log_info(msg="CUDA is not available, using CPU instead.", level="WARNING", state=False, logger=logger):
                device = torch.device("cpu")
        if len(gpu_list) == 0:
            with log_info(msg="Use CPU.", level="INFO", state=False, logger=logger):
                device = torch.device("cpu")
        elif len(gpu_list) == 1:
            with log_info(msg="Use GPU {}.".format(gpu_list[0]), level="INFO", state=False, logger=logger):
                device = torch.device("cuda:{}".format(gpu_list[0]))
                model = model.to(device)
        elif len(gpu_list) > 1:
            raise NotImplementedError("Multi-GPU mode is not implemented yet.")
    return model, device

def try_make_path_exists(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except:
            return False
    return True

def save_ckpt(path2file, logger=None, **ckpt):
    with log_info(msg="Save checkpoint to {}".format(path2file), level="INFO", state=True, logger=logger):
        torch.save(ckpt, path2file)

def update_best_ckpt(path2file, metrics_logger, logger=None, **ckpt):
    pass

def pack_code(cfg, logger=None):
    src_dir = cfg.GENERAL.ROOT
    src_items = [
        "src"
    ]
    des_dir = cfg.LOG.DIR
    with log_info(msg="Pack items {} from ROOT to {}".format(src_items, des_dir), level="INFO", state=True, logger=logger):
        t = time.gmtime()
        for item in src_items:
            path2src = os.path.join(src_dir, item)
            path2des = os.path.join("{}/{}/Mon{}Day{}Hour{}Min{}".format(
                des_dir, 
                "src", 
                str(t.tm_mon).zfill(2), 
                str(t.tm_mday).zfill(2), 
                str(t.tm_hour).zfill(2), 
                str(t.tm_min).zfill(2), 
            ))
            try_make_path_exists(path2des)
            os.system("cp -r {} {}".format(path2src, path2des))
    # raise NotImplementedError("Function pack_code is not implemented yet.")


if __name__ == "__main__":
    log_info(msg="DEBUG MESSAGE", level="DEBUG", state=False, logger=None)
    