
"""
Author:
    Yiqun Chen
Docs:
    Main functition to run program.
"""

import sys, os, copy
import torch, torchvision

from configs.configs import cfg
from utils import utils, loss_fn_helper, lr_scheduler_helper, optimizer_helper
from utils.logger import Logger
from models import model_builder
from data import data_loader
from train import train_one_epoch
from evaluate import evaluate
from generate import generate
from utils.metrics import Metrics

def main():
    # Set logger to record information.
    logger = Logger(cfg)
    logger.log_info(cfg)
    metrics_logger = Metrics()
    utils.pack_code(cfg, logger=logger)

    # Build model.
    model = model_builder.build_model(cfg=cfg, logger=logger)

    # Read checkpoint.
    ckpt = torch.load(cfg.MODEL.PATH2CKPT) if cfg.GENERAL.RESUME else {}

    if cfg.GENERAL.RESUME:
        model.load_state_dict(ckpt["model"])
    resume_epoch = ckpt["epoch"] if cfg.GENERAL.RESUME else 0
    optimizer = ckpt["optimizer"] if cfg.GENERAL.RESUME else optimizer_helper.build_optimizer(cfg=cfg, model=model)
    lr_scheduler = ckpt["lr_scheduler"] if cfg.GENERAL.RESUME else lr_scheduler_helper.build_scheduler(cfg=cfg, optimizer=optimizer)
    # lr_scheduler = lr_scheduler_helper.build_scheduler(cfg=cfg, optimizer=optimizer)
    # lr_scheduler.sychronize(resume_epoch)
    loss_fn = ckpt["loss_fn"] if cfg.GENERAL.RESUME else loss_fn_helper.build_loss_fn(cfg=cfg)
    
    # Set device.
    model, device = utils.set_device(model, cfg.GENERAL.GPU)
    
    # Prepare dataset.
    if cfg.GENERAL.TRAIN:
        try:
            train_data_loader = data_loader.build_data_loader(cfg, cfg.DATA.DATASET, "train")
        except:
            logger.log_info("Cannot build train dataset.")
    if cfg.GENERAL.VALID:
        try:
            valid_data_loader = data_loader.build_data_loader(cfg, cfg.DATA.DATASET, "valid")
        except:
            logger.log_info("Cannot build valid dataset.")
    if cfg.GENERAL.TEST:
        try:
            test_data_loader = data_loader.build_data_loader(cfg, cfg.DATA.DATASET, "test")
        except:
            logger.log_info("Cannot build test dataset.")

    # ################ NOTE DEBUG NOTE ################
    
    '''train_one_epoch(
        epoch=0,
        cfg=cfg,  
        model=model, 
        data_loader=train_data_loader, 
        device=device, 
        loss_fn=loss_fn, 
        optimizer=optimizer, 
        lr_scheduler=lr_scheduler, 
        metrics_logger=metrics_logger, 
        logger=logger, 
    )'''
    '''
    generate(
        cfg=cfg, 
        model=model, 
        data_loader=test_data_loader, 
        device=device, 
        phase="test", 
        logger=logger, 
    )'''
    
    # ################ NOTE DEBUG NOTE ################

    # Train, evaluate model and save checkpoint.
    for epoch in range(cfg.TRAIN.MAX_EPOCH):
        if not cfg.GENERAL.TRAIN:
            break
        if resume_epoch > epoch:
            continue

        try:
            train_one_epoch(
                epoch=epoch,
                cfg=cfg,  
                model=model, 
                data_loader=train_data_loader, 
                device=device, 
                loss_fn=loss_fn, 
                optimizer=optimizer, 
                lr_scheduler=lr_scheduler, 
                metrics_logger=metrics_logger, 
                logger=logger, 
            )
        except:
            logger.log_info("Failed to train model.")
        
        optimizer.zero_grad()
        with torch.no_grad():
            utils.save_ckpt(
                path2file=os.path.join(cfg.MODEL.CKPT_DIR, cfg.GENERAL.ID + "_" + str(epoch).zfill(3) + ".pth"), 
                logger=logger, 
                model=model.state_dict(), 
                epoch=epoch, 
                optimizer=optimizer, 
                lr_scheduler=lr_scheduler, # NOTE Need attribdict>=0.0.5
                loss_fn=loss_fn, 
                metrics=metrics_logger, 
            )
        try:
            evaluate(
                epoch=epoch, 
                cfg=cfg, 
                model=model, 
                data_loader=valid_data_loader, 
                device=device, 
                loss_fn=loss_fn, 
                metrics_logger=metrics_logger, 
                phase="valid", 
                logger=logger,
                save=cfg.SAVE.SAVE,  
            )
        except:
            logger.log_info("Failed to evaluate model.")

        with torch.no_grad():
            utils.save_ckpt(
                path2file=os.path.join(cfg.MODEL.CKPT_DIR, cfg.GENERAL.ID + "_" + str(epoch).zfill(3) + ".pth"), 
                logger=logger, 
                model=model.state_dict(), 
                epoch=epoch, 
                optimizer=optimizer, 
                lr_scheduler=lr_scheduler, # NOTE Need attribdict>=0.0.5
                loss_fn=loss_fn, 
                metrics=metrics_logger, 
            )

    # If test set has target images, evaluate and save them, otherwise just try to generate output images.
    if cfg.DATA.DATASET == "DualPixelNTIRE2021":
        try:
            generate(
                cfg=cfg,
                model=model, 
                data_loader=valid_data_loader, 
                device=device, 
                phase="valid", 
                logger=logger, 
            )
        except:
            logger.log_info("Failed to generate output images of valid set of NTIRE2021.")
    try:
        evaluate(
            epoch=epoch, 
            cfg=cfg, 
            model=model, 
            data_loader=test_data_loader, 
            device=device, 
            loss_fn=loss_fn, 
            metrics_logger=metrics_logger, 
            phase="test", 
            logger=logger, 
            save=True, 
        )
    except:
        logger.log_info("Failed to test model, try to generate images.")
        try:
            generate(
                cfg=cfg, 
                model=model, 
                data_loader=test_data_loader, 
                device=device, 
                phase="test", 
                logger=logger, 
            )
        except:
            logger.log_info("Cannot generate output images of test set.")
    return None


if __name__ == "__main__":
    main()


