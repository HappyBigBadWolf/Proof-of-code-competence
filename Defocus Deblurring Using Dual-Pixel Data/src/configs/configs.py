
"""
Author:
    Yiqun Chen
Docs:
    Configurations, should not call other custom modules.
"""

import os, sys, copy, argparse
from alphaconfig import AlphaConfig

configs = AlphaConfig()
cfg = configs

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=str, required=True)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--resume", default="false", choices=["true", "false"], type=str, required=True)
parser.add_argument("--train", default="true", choices=["true", "false"], type=str, required=True)
parser.add_argument("--valid", default="true", choices=["true", "false"], type=str, required=True)
parser.add_argument("--test", default="false", choices=["true", "false"], type=str, required=True)
parser.add_argument("--gpu", type=str, required=True)
args = parser.parse_args()

# ================================ 
# GENERAL
# ================================ 
cfg.GENERAL.ROOT                                =   os.path.join(os.getcwd(), ".")
cfg.GENERAL.ID                                  =   "{}".format(args.id)
cfg.GENERAL.BATCH_SIZE                          =   args.batch_size
cfg.GENERAL.TRAIN                               =   True if args.train == "true" else False
cfg.GENERAL.VALID                               =   True if args.valid == "true" else False
cfg.GENERAL.TEST                                =   True if args.test == "true" else False
cfg.GENERAL.RESUME                              =   True if args.resume == "true" else False
cfg.GENERAL.GPU                                 =   eval(args.gpu)

# ================================ 
# MODEL
# ================================ 
cfg.MODEL.ARCH                                  =   None # TODO
cfg.MODEL.ENCODER                               =   "EGEncoderV1" # ["DPDEncoder", "DPDEncoderV2"]
cfg.MODEL.DECODER                               =   "DPDDecoder" # ["DPDDecoder", "DPDDecoderV2"]
cfg.MODEL.CKPT_DIR                              =   os.path.join(cfg.GENERAL.ROOT, "checkpoints", cfg.GENERAL.ID)
# cfg.MODEL.PATH2CKPT                             =   os.path.join(cfg.MODEL.CKPT_DIR, sorted(os.listdir(cfg.MODEL.CKPT_DIR))[-1]) if cfg.GENERAL.RESUME else cfg.MODEL.CKPT_DIR
cfg.MODEL.PATH2CKPT                             =   os.path.join(cfg.MODEL.CKPT_DIR, "{}.pth".format(cfg.GENERAL.ID))
cfg.MODEL.BOTTLENECK                            =   "DPDBottleneck"

# ================================ 
# DATA
# ================================ 
cfg.DATA.DIR                                    =   {
    "DualPixelNTIRE2021": "/home/yqchen/Data/DualPixelNTIRE2021", 
    # "DualPixelNTIRE2021": "/mnt/g/Datasets/DualPixelNTIRE2021", 
    "DualPixelCanon": "/home/yqchen/Data/DualPixelCanon", 
}
cfg.DATA.NUMWORKERS                             =   4 
cfg.DATA.DATASET                                =   args.dataset # "DualPixelCanon"
cfg.DATA.BIT_DEPTH                              =   16 # NOTE DO NOT CHANGE THIS
cfg.DATA.MEAN                                   =   [0, 0, 0]
cfg.DATA.NORM                                   =   [2**cfg.DATA.BIT_DEPTH-1, 2**cfg.DATA.BIT_DEPTH-1, 2**cfg.DATA.BIT_DEPTH-1]
cfg.DATA.AUGMENTATION                           =   True
cfg.DATA.RANDOM_SAMPLE_RATIO                    =   0.4

# ================================ 
# OPTIMIZER
# ================================ 
cfg.OPTIMIZER.OPTIMIZER                         =   "Adam" # ["Adam", "SGD", "AdamW"]
cfg.OPTIMIZER.LR                                =   2e-5 # * cfg.GENERAL.BATCH_SIZE / 5

# ================================ 
# SCHEDULER
# ================================ 
cfg.SCHEDULER.SCHEDULER                         =   "StepLRScheduler"
cfg.SCHEDULER.UPDATE_EPOCH                      =   range(60, 200, 60) # at which epoch to decay/update learning rate.
cfg.SCHEDULER.UPDATE_SCALE                      =   0.5 # decay ratio of learning rate.

# ================================ 
# SCHEDULER
# ================================ 
cfg.TRAIN.MAX_EPOCH                             =   200
cfg.TRAIN.RANDOM_SAMPLE_RATIO                   =   0.3 # ratio of samples used in training phase.

# ================================ 
# LOSS_FN
# ================================ 
cfg.LOSS_FN.LOSS_FN                             =   "MSESSIMLoss" # see src/utils/loss_fn_helper.py for more details.
cfg.LOSS_FN.WEIGHTS                             =   {
    "L1SPAT": 1.0, "L2SPAT": 1.0, "L1FREQ": 1.0, "L2FREQ": 1.0, "LPIPS": 1.0, "SSIM": 0.3, 
} # weights of different losses.

# ================================ 
# LOG
# ================================ 
cfg.SAVE.DIR                                    =   os.path.join(os.path.join(cfg.GENERAL.ROOT, "results", cfg.GENERAL.ID, cfg.DATA.DATASET))
cfg.SAVE.SAVE                                   =   True

# ================================ 
# LOG
# ================================ 
cfg.LOG.DIR                                     =   os.path.join(os.path.join(cfg.GENERAL.ROOT, "logs", cfg.GENERAL.ID))

cfg.cvt_state(read_only=True)
    
_paths = [
    cfg.LOG.DIR, 
    cfg.MODEL.CKPT_DIR, 
    cfg.SAVE.DIR, 
]
_paths.extend(list(cfg.DATA.DIR.values()))

for _path in _paths:
    if not os.path.exists(_path):
        os.makedirs(_path)

# raise NotImplementedError("Please set your configurations and remove this error message.")
