
"""
Author:
    Yiqun Chen
Docs:
    Configurations, should not call other custom modules.
"""

import os, sys, copy, argparse
from attribdict import AttribDict as Dict

configs = Dict()
cfg = configs

parser = argparse.ArgumentParser()
parser.add_argument("id", type=str)
parser.add_argument("strict_id", default="true", choices=["true", "false"], type=str)
parser.add_argument("batch_size", type=int)
parser.add_argument("train", default="true", choices=["true", "false"], type=str)
parser.add_argument("valid", default="true", choices=["true", "false"], type=str)
parser.add_argument("test", default="false", choices=["true", "false"], type=str)
parser.add_argument("resume", default="false", choices=["true", "false"], type=str)
parser.add_argument("tiny_dataset", default="false", choices=["true", "false"], type=str)
parser.add_argument("gpu", type=str)
args = parser.parse_args()

# ================================ 
# GENERAL
# ================================ 
cfg.GENERAL.ROOT                                =   os.path.join(os.getcwd(), ".")
cfg.GENERAL.ID                                  =   "{}".format(args.id)
cfg.GENERAL.STRICT_ID                           =   True if args.strict_id == "true" else False
cfg.GENERAL.BATCH_SIZE                          =   args.batch_size
cfg.GENERAL.TRAIN                               =   True if args.train == "true" else False
cfg.GENERAL.VALID                               =   True if args.valid == "true" else False
cfg.GENERAL.TEST                                =   True if args.test == "true" else False
cfg.GENERAL.RESUME                              =   True if args.resume == "true" else False
cfg.GENERAL.TINY_DATASET                        =   True if args.tiny_dataset == "true" else False
cfg.GENERAL.GPU                                 =   eval(args.gpu)

# ================================ 
# MODEL
# ================================ 
cfg.MODEL.ARCH                                  =   None # TODO
cfg.MODEL.ENCODER                               =   "DPDEncoderV2" # ["DPDEncoder", "DPDEncoderV2"]
cfg.MODEL.DECODER                               =   "DPDDecoderV2" # ["DPDDecoder", "DPDDecoderV2"]
cfg.MODEL.CKPT_DIR                              =   os.path.join(cfg.GENERAL.ROOT, "checkpoints", cfg.GENERAL.ID)
cfg.MODEL.PATH2CKPT                             =   os.path.join(cfg.MODEL.CKPT_DIR, sorted(os.listdir(cfg.MODEL.CKPT_DIR))[-1]) if cfg.GENERAL.RESUME else cfg.MODEL.CKPT_DIR
# cfg.MODEL.PATH2CKPT                             =   os.path.join(cfg.MODEL.CKPT_DIR, "03_099.pth") 

# ================================ 
# DATA
# ================================ 
cfg.DATA.DIR                                    =   {
    "DualPixelNTIRE2021": "/home/yqchen/Data/DualPixelNTIRE2021", 
    # "DualPixelNTIRE2021": "/mnt/g/Datasets/DualPixelNTIRE2021", 
    "DualPixelCanon": "/home/yqchen/Data/DualPixelCanon", 
}
cfg.DATA.NUMWORKERS                             =   4 # eargs.batch_size
cfg.DATA.DATASET                                =   "DualPixelCanon"
cfg.DATA.BIT_DEPTH                              =   16
cfg.DATA.MEAN                                   =   [0, 0, 0]
cfg.DATA.NORM                                   =   [2**cfg.DATA.BIT_DEPTH-1, 2**cfg.DATA.BIT_DEPTH-1, 2**cfg.DATA.BIT_DEPTH-1]
# cfg.DATA.PREFETCH                               =   False # It seems helpless.
cfg.DATA.AUGMENTATION                           =   True
cfg.DATA.CHOICE_RATIO                           =   0.3

# ================================ 
# OPTIMIZER
# ================================ 
cfg.OPTIMIZER.OPTIMIZER                         =   "Adam" # ["Adam", "SGD", "AdamW"]
cfg.OPTIMIZER.LR                                =   2e-5 # * cfg.GENERAL.BATCH_SIZE / 5

# ================================ 
# SCHEDULER
# ================================ 
cfg.SCHEDULER.SCHEDULER                         =   "StepLRScheduler"
cfg.SCHEDULER.UPDATE_EPOCH                      =   range(60, 200, 60)
cfg.SCHEDULER.UPDATE_SCALE                      =   0.5

# ================================ 
# SCHEDULER
# ================================ 
cfg.TRAIN.MAX_EPOCH                             =   200
cfg.TRAIN.RANDOM_SAMPLE_RATIO                   =   0.3

# ================================ 
# LOSS_FN
# ================================ 
cfg.LOSS_FN.LOSS_FN                             =   "MSELoss"
cfg.LOSS_FN.MSEMAE_WEIGHT                       =   {"MSE": 1.0, "MAE": 1.0}

# ================================ 
# LOG
# ================================ 
cfg.SAVE.DIR                                    =   os.path.join(os.path.join(cfg.GENERAL.ROOT, "results", cfg.GENERAL.ID, cfg.DATA.DATASET))
cfg.SAVE.SAVE                                   =   True

# ================================ 
# LOG
# ================================ 
cfg.LOG.DIR                                     =   os.path.join(os.path.join(cfg.GENERAL.ROOT, "logs", cfg.GENERAL.ID))

if cfg.GENERAL.STRICT_ID:
    assert not os.path.exists(cfg.LOG.DIR), "Cannot use same ID in strict mode."
    
_paths = [
    cfg.LOG.DIR, 
    cfg.MODEL.CKPT_DIR, 
    cfg.SAVE.DIR, 
]
_paths.extend(list(cfg.DATA.DIR.as_dict().values()))

for _path in _paths:
    if not os.path.exists(_path):
        os.makedirs(_path)

# raise NotImplementedError("Please set your configurations and remove this error message.")
