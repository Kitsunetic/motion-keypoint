from pathlib import Path

import yaml
from easydict import EasyDict

import utils


def load_config(config_file):
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

    if config.inference:
        log = utils.CustomLogger(config.result_dir / f"{config.uid}-inference.log", "a")
    else:
        log = utils.CustomLogger(config.result_dir / f"{config.uid}.log", "a")
    log.file.write("\r\n")
    log.info("학습 시작")
    for key, value in config.items():
        log.info(f"{key}: {value}")
    log.flush()

    utils.seed_everything(config.seed, deterministic=False)
    config.log = log
    return config
