import argparse
from pprint import pformat
import json
import math
import os
from random import random, randint
import shutil
import sys
from multiprocessing import cpu_count
from pathlib import Path

import imageio
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from easydict import EasyDict
from PIL import Image
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm

import networks
import utils
from datasets import get_det_dataset


class DetTrainOutput:
    def __init__(self):
        self.loss = utils.AverageMeter()

    def freeze(self):
        self.loss = self.loss()
        return self


class DetTrainer:
    _tqdm_ = dict(ncols=100, leave=False, file=sys.stdout)

    def __init__(self, C, fold=1, checkpoint=None):
        self.C = C
        self.fold = fold

        self.det_model = networks.EfficientDet(self.C.det_model.name, pretrained=True)
        self.det_model.cuda()

        # Optimizer
        if self.C.train.SAM:
            self.optimizer = utils.SAM(self.det_model.parameters(), optim.AdamW, lr=self.C.train.lr)
        else:
            self.optimizer = optim.AdamW(self.det_model.parameters(), lr=self.C.train.lr)

        self.epoch = self.C.train.start_epoch
        self.best_loss = math.inf
        self.earlystop_cnt = 0

        # Dataset
        self.dl_train, self.dl_valid = get_det_dataset(C, self.fold)

        # Load Checkpoint
        if checkpoint is not None:
            self.load(checkpoint)

        # Scheduler
        if self.C.train.scheduler.type == "CosineAnnealingWarmUpRestarts":
            self.scheduler = utils.CosineAnnealingWarmUpRestarts(
                self.optimizer, **self.C.train.scheduler.params, last_epoch=self.epoch - 2
            )
        elif self.C.train.scheduler.type == "CosineAnnealingWarmRestarts":
            self.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, **self.C.train.scheduler.params, last_epoch=self.epoch - 2
            )
        elif self.C.train.scheduler.type == "ReduceLROnPlateau":
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, **self.C.train.scheduler.params)

    def save(self, path):
        torch.save(
            {
                "model": self.det_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": self.epoch,
                "best_loss": self.best_loss,
                "earlystop_cnt": self.earlystop_cnt,
            },
            path,
        )

    def load(self, path):
        print("Load pretrained", path)
        ckpt = torch.load(path)
        self.det_model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epoch = ckpt["epoch"] + 1
        self.best_loss = ckpt["best_loss"]
        self.earlystop_cnt = ckpt["earlystop_cnt"]

    def close(self):
        self.logger.close()

    def train_loop(self):
        self.det_model.train()

        O = DetTrainOutput()
        with tqdm(total=len(self.dl_train.dataset), **self._tqdm_, desc=f"Train {self.epoch:03d}") as t:
            for files, imgs, annots in self.dl_train:
                imgs_, annots_ = imgs.cuda(non_blocking=True), annots.cuda(non_blocking=True)

                # batch augmentation
                if self.C.train.batch_augmentation:
                    h, w = imgs.shape[2:]

                    # downsample
                    if random() <= 0.5:
                        imgs_ = F.interpolate(imgs_, (h // 2, w // 2))
                        annots_[:4] *= 0.5

                    # rotation
                    if random() <= 0.5:
                        k = randint(1, 3)
                        a, b, c, d = annots_[..., 0], annots_[..., 1], annots_[..., 2], annots_[..., 3]
                        e = annots_[..., 4]
                        if k == 1:
                            annots_ = torch.stack([b, w - c, d, w - a, e], dim=2)
                        elif k == 2:
                            annots_ = torch.stack([w - c, h - d, w - a, h - b, e], dim=2)
                        elif k == 3:
                            annots_ = torch.stack([h - d, a, h - b, c, e], dim=2)
                        imgs_ = torch.rot90(imgs_, k=k, dims=(2, 3))

                loss = self.det_model(imgs_, annots_)

                self.optimizer.zero_grad()
                loss.backward()
                if isinstance(self.optimizer, utils.SAM):
                    self.optimizer.first_step()
                    self.det_model(imgs_, annots_).backward()
                    self.optimizer.second_step()
                else:
                    self.optimizer.step()

                O.loss.update(loss.item(), len(files))
                t.set_postfix_str(f"loss: {loss.item():.6f}", refresh=False)
                t.update(len(files))

        return O.freeze()

    @torch.no_grad()
    def valid_loop(self):
        self.det_model.eval()

        O = DetTrainOutput()
        with tqdm(total=len(self.dl_valid.dataset), **self._tqdm_, desc=f"Valid {self.epoch:03d}") as t:
            for files, imgs, annots in self.dl_valid:
                imgs_, annots_ = imgs.cuda(non_blocking=True), annots.cuda(non_blocking=True)
                loss = self.det_model(imgs_, annots_)

                O.loss.update(loss.item(), len(files))
                t.set_postfix_str(f"loss: {loss.item():.6f}", refresh=False)
                t.update(len(files))

        return O.freeze()

    @torch.no_grad()
    def callback(self, to: DetTrainOutput, vo: DetTrainOutput):
        self.C.log.info(
            f"Epoch: {self.epoch:03d},",
            f"loss: {to.loss:.6f};{vo.loss:.6f},",
        )
        self.C.log.flush()

        if isinstance(self.scheduler, lr_scheduler.CosineAnnealingWarmRestarts):
            self.scheduler.step()
        elif isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(vo.loss)

        if self.best_loss > vo.loss:
            self.best_loss = vo.loss
            self.earlystop_cnt = 0
            self.save(self.C.result_dir / f"{self.C.uid}_{self.fold}.pth")
        else:
            self.earlystop_cnt += 1

    def fit(self):
        for self.epoch in range(self.epoch, self.C.train.final_epoch + 1):
            to = self.train_loop()
            vo = self.valid_loop()
            self.callback(to, vo)

            if self.earlystop_cnt > self.C.train.earlystop_patience:
                self.C.log.info(f"Stop training at epoch", self.epoch)
                break

        self.load(self.C.result_dir / f"{self.C.uid}_{self.fold}.pth")


def main():
    args = argparse.ArgumentParser()
    args.add_argument("config_file", type=str)
    args = args.parse_args(sys.argv[1:])

    with open(args.config_file, "r") as f:
        C = EasyDict(yaml.load(f, yaml.FullLoader))

    for fold, checkpoint in zip(C.train.folds, C.train.checkpoints):
        with open(args.config_file, "r") as f:
            C = EasyDict(yaml.load(f, yaml.FullLoader))
            Path(C.result_dir).mkdir(parents=True, exist_ok=True)

            if C.dataset.num_cpus < 0:
                C.dataset.num_cpus = cpu_count()
            C.uid = f"{C.det_model.name}"
            C.uid += "-sam" if C.train.SAM else ""
            C.uid += f"-{C.dataset.input_width}x{C.dataset.input_height}"
            C.uid += f"-pad{C.dataset.padding}"
            C.uid += "-ba" if C.train.batch_augmentation else ""
            C.uid += f"-{C.comment}" if C.comment is not None else ""
            C.uid += f"_{fold}"

            log = utils.CustomLogger(Path(C.result_dir) / f"{C.uid}_{fold}.log", "a")
            log.file.write("\r\n\r\n")
            log.info("\r\n" + pformat(C))
            log.flush()

            C.log = log
            C.result_dir = Path(C.result_dir)
            C.dataset.dir = Path(C.dataset.dir)
            utils.seed_everything(C.seed)

        trainer = DetTrainer(C, fold, checkpoint)
        trainer.fit()


if __name__ == "__main__":
    main()
