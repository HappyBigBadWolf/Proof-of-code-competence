
"""
Author:
    Yiqun Chen
Docs:
    Dataset classes.
"""

import os, sys, random
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
# PIL.Image seems cannot read 16-bit .png file.
# from PIL import Image
import cv2, copy
from tqdm import tqdm
import numpy as np

from utils import utils

_DATASET = {}

def add_dataset(dataset):
    _DATASET[dataset.__name__] = dataset
    return dataset


@add_dataset
class DualPixelCanon(torch.utils.data.Dataset):
    """
    Info:
        Dataset of Canon, folders and files structure:
        DualPixelCanon:
            - train:
                - source:
                    00000.png
                    00001.png
                    ...
                - target:
                    00000.png
                    00001.png
                    ...
                - l_view:
                    00000.png
                    00001.png
                    ...
                - r_view:
                    00000.png
                    00001.png
                    ...
            - valid:
                ...
            - test:
                ...
    """
    def __init__(self, cfg, split, *args, **kwargs):
        super(DualPixelCanon, self).__init__()
        self.cfg = cfg
        self.split = split
        self._build()

    def _build(self):
        self.data = []
        self.t_img_list = []
        self.s_img_list = []
        self.l_img_list = []
        self.r_img_list = []
        self.path2imgs = os.path.join(self.cfg.DATA.DIR[self.cfg.DATA.DATASET], self.split)

        t_img_list = np.array(sorted(os.listdir(os.path.join(self.path2imgs, "target"))))
        s_img_list = np.array(sorted(os.listdir(os.path.join(self.path2imgs, "source"))))
        l_img_list = np.array(sorted(os.listdir(os.path.join(self.path2imgs, "l_view"))))
        r_img_list = np.array(sorted(os.listdir(os.path.join(self.path2imgs, "r_view"))))

        if self.cfg.GENERAL.TINY_DATASET and self.split == "train":
            k = int(len(t_img_list)*self.cfg.DATA.CHOICE_RATIO)
            choices = sorted(random.sample(range(0, len(t_img_list)), k))
            t_img_list = t_img_list[choices]
            s_img_list = s_img_list[choices]
            l_img_list = l_img_list[choices]
            r_img_list = r_img_list[choices]

        t_img_dict = {img.split(".")[0]: img for img in t_img_list}
        s_img_dict = {img.split(".")[0]: img for img in s_img_list}
        l_img_dict = {img.split(".")[0]: img for img in l_img_list}
        r_img_dict = {img.split(".")[0]: img for img in r_img_list}

        pbar = tqdm(total=len(t_img_list), dynamic_ncols=True)
        for idx, img_idx in enumerate(t_img_dict.keys()):
            if img_idx == "":
                continue
            if img_idx in l_img_dict.keys() and img_idx in r_img_dict.keys():
                self.data.append({
                    "img_idx": img_idx, 
                    "target": t_img_dict[img_idx], 
                    "source": s_img_dict[img_idx], 
                    "l_view": l_img_dict[img_idx], 
                    "r_view": r_img_dict[img_idx], 
                })

            pbar.update()
        pbar.close()
        # self._preprocess()
        # raise NotImplementedError("Method DualPixelCanon._build is not implemented yet.")

    def _preprocess(self):
        raise NotImplementedError("Method DualPixelCanon._preprocess is not implemented yet.")

    def __getitem__(self, idx):
        data = {}
        
        if self.split == "train" and self.cfg.DATA.AUGMENTATION and (np.random.randn() < 0.5):
            target = cv2.imread(os.path.join(self.path2imgs, "target", self.data[idx]["target"]), -1)
            source = copy.deepcopy(target)
            l_img = copy.deepcopy(target)
            r_img = copy.deepcopy(target)
        else:
            target = cv2.imread(os.path.join(self.path2imgs, "target", self.data[idx]["target"]), -1)
            source = cv2.imread(os.path.join(self.path2imgs, "source", self.data[idx]["source"]), -1)
            l_img = cv2.imread(os.path.join(self.path2imgs, "l_view", self.data[idx]["l_view"]), -1)
            r_img = cv2.imread(os.path.join(self.path2imgs, "r_view", self.data[idx]["r_view"]), -1)

        data["img_idx"] = self.data[idx]["img_idx"]
        # Transform: [H, W, C] -> [C, H, W]
        data["target"] = np.transpose((target-np.array(self.cfg.DATA.MEAN))/np.array(self.cfg.DATA.NORM), (2, 0, 1)).astype(np.float32)
        data["source"] = np.transpose((source-np.array(self.cfg.DATA.MEAN))/np.array(self.cfg.DATA.NORM), (2, 0, 1)).astype(np.float32)
        data["l_view"] = np.transpose((l_img-np.array(self.cfg.DATA.MEAN))/np.array(self.cfg.DATA.NORM), (2, 0, 1)).astype(np.float32)
        data["r_view"] = np.transpose((r_img-np.array(self.cfg.DATA.MEAN))/np.array(self.cfg.DATA.NORM), (2, 0, 1)).astype(np.float32)
        
        return data

    def __len__(self):
        return len(self.data)


@add_dataset
class DualPixelNTIRE2021(torch.utils.data.Dataset):
    """
    Info:
        Dataset of NTIRE2021, folders and files structure:
        DualPixelNTIRE2021:
            - train:
                - target:
                    00000.png
                    00001.png
                    ...
                - l_view:
                    00000.png
                    00001.png
                    ...
                - r_view:
                    00000.png
                    00001.png
                ...
            - valid:
                ...
            - test:
                ...
    """
    def __init__(self, cfg, split, *args, **kwargs):
        super(DualPixelNTIRE2021, self).__init__()
        self.cfg = cfg
        self.split = split
        self._build()

    def _build(self):
        self.data = []
        self.t_img_list = []
        self.s_img_list = []
        self.l_img_list = []
        self.r_img_list = []
        self.path2imgs = os.path.join(self.cfg.DATA.DIR[self.cfg.DATA.DATASET], self.split)

        t_img_list = os.listdir(os.path.join(self.path2imgs, "target"))
        r_img_list = os.listdir(os.path.join(self.path2imgs, "r_view"))
        l_img_list = os.listdir(os.path.join(self.path2imgs, "l_view"))

        if self.cfg.GENERAL.TINY_DATASET and self.split == "train":
            k = int(len(t_img_list) * self.cfg.DATA.CHOICE_RATIO)
            choices = sorted(random.sample(range(0, len(t_img_list)), k))
            t_img_list = t_img_list[choices]
            l_img_list = l_img_list[choices]
            r_img_list = r_img_list[choices]

        t_img_dict = {img.split(".")[0]: img for img in t_img_list}
        l_img_dict = {img.split(".")[0]: img for img in l_img_list}
        r_img_dict = {img.split(".")[0]: img for img in r_img_list}

        pbar = tqdm(total=len(t_img_list), dynamic_ncols=True)
        for idx, img_idx in enumerate(l_img_dict.keys()):
            if img_idx == "":
                continue
            if img_idx in l_img_dict.keys() and img_idx in r_img_dict.keys():
                item = {
                    "img_idx": img_idx, 
                    "l_view": l_img_dict[img_idx], 
                    "r_view": r_img_dict[img_idx], 
                }
                if self.split in ["train"]:
                    item["target"] = t_img_dict[img_idx]
                self.data.append(item)

            pbar.update()
        pbar.close()

    def _preprocess(self):
        raise ValueError("Some images in training set is broken, please rectify them.")
        raise NotImplementedError("Method DualPixelNTIRE2021._preprocess is not implemented yet.")

    def __getitem__(self, idx):
        data = {}

        if self.split == "train" and self.cfg.DATA.AUGMENTATION and (np.random.randn() < 0.5):
            target = cv2.imread(os.path.join(self.path2imgs, "target", self.data[idx]["target"]), -1)
            l_img = copy.deepcopy(target)
            r_img = copy.deepcopy(target)
        else:
            if self.split == "train":
                target = cv2.imread(os.path.join(self.path2imgs, "target", self.data[idx]["target"]), -1)
                data["target"] = np.transpose((target-np.array(self.cfg.DATA.MEAN))/np.array(self.cfg.DATA.NORM), (2, 0, 1)).astype(np.float32)
            l_img = cv2.imread(os.path.join(self.path2imgs, "l_view", self.data[idx]["l_view"]), -1)
            r_img = cv2.imread(os.path.join(self.path2imgs, "r_view", self.data[idx]["r_view"]), -1)

        data["img_idx"] = self.data[idx]["img_idx"]
        # Transform: [H, W, C] -> [C, H, W]
        
        data["l_view"] = np.transpose((l_img-np.array(self.cfg.DATA.MEAN))/np.array(self.cfg.DATA.NORM), (2, 0, 1)).astype(np.float32)
        data["r_view"] = np.transpose((r_img-np.array(self.cfg.DATA.MEAN))/np.array(self.cfg.DATA.NORM), (2, 0, 1)).astype(np.float32)

        return data
        
    def __len__(self):
        return len(self.data)
        

@add_dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split, *args, **kwargs):
        super(Dataset, self).__init__()
        self.cfg = cfg
        self.split = split
        self._build()

    def _build(self):
        raise NotImplementedError("Dataset is not implemeted yet.")

    def __len__(self):
        raise NotImplementedError("Dataset is not implemeted yet.")

    def __getitem__(self, idx):
        raise NotImplementedError("Dataset is not implemeted yet.")


if __name__ == "__main__":
    from configs.configs import cfg

    dataset = DualPixelNTIRE2021(cfg, "train")
    for i in range(len(dataset)):
        item = dataset[i]
        