import os
from pathlib import Path
from pprint import pformat

import yaml
from easydict import EasyDict
from torch.optim import lr_scheduler

import utils


def load_config(config_file, write_log=True):
    with open(config_file, "r") as f:
        config = yaml.load(f, yaml.FullLoader)
        config = EasyDict(config)

    config.result_dir = Path(config.result_dir)
    config.result_dir.mkdir(parents=True, exist_ok=True)
    config.data_dir = Path(config.data_dir)

    if config.debug:
        config.step1_epoch = 2
        config.step2_epoch = 4
        config.step3_epoch = 10

    if write_log:
        log = utils.CustomLogger(config.result_dir / f"{config.uid}.log", "a")
        log.file.write("\r\n\r\n\r\n")
        log.info("===============================================================")
        log.info("학습 시작")
        log.info("\r\n" + pformat(config))
        log.flush()
    else:
        log = utils.CustomLogger_(config.result_dir / f"{config.uid}.log", "a")

    utils.seed_everything(config.seed, deterministic=False)
    config.log = log
    return config


def load_config_effdet(config_file, write_log=True):
    with open(config_file, "r") as f:
        config = yaml.load(f, yaml.FullLoader)
        config = EasyDict(config)

    if config.debug:
        config.final_epoch = 4

    log_path = os.path.join(config.result_dir, f"{config.uid}.log")
    if write_log:
        log = utils.CustomLogger(config.result_dir / f"{config.uid}.log", "a")
        log.file.write("\r\n\r\n\r\n")
        log.info("===============================================================")
        log.info("학습 시작")
        log.info("\r\n" + pformat(config))
        log.flush()
    else:
        log = utils.CustomLogger_(config.result_dir / f"{config.uid}.log", "a")

    config.result_dir = Path(config.result_dir)
    config.result_dir.mkdir(parents=True, exist_ok=True)
    config.data_dir = Path(config.data_dir)

    utils.seed_everything(config.seed, deterministic=False)
    config.log = log
    return config


def get_scheduler(config, optimizer, last_epoch):
    if config.scheduler.type == "CosineAnnealingWarmUpRestarts":
        return utils.CosineAnnealingWarmUpRestarts(
            optimizer,
            **config.scheduler.params,
            last_epoch=last_epoch,
        )
    elif config.scheduler.type == "CosineAnnealingWarmRestarts":
        return lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            **config.scheduler.params,
            last_epoch=last_epoch,
        )
    elif config.scheduler.type == "ReduceLROnPlateau":
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            **config.scheduler.params,
        )
