import argparse
import json
import math
import multiprocessing
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat

import cv2
import imageio
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from easydict import EasyDict
from PIL import Image
from sklearn.model_selection import KFold
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

import networks
import utils
from datasets import get_pose_datasets
from losses import AWing, JointMSELoss, KeypointBCELoss, KeypointLoss, KeypointRMSE, SigmoidKLDivLoss, SigmoidMAE


class TrainOutput:
    def __init__(self):
        self.loss = utils.AverageMeter()
        self.rmse = utils.AverageMeter()

    def freeze(self):
        self.loss = self.loss()
        self.rmse = self.rmse()
        return self


class PoseTrainer:
    _tqdm_ = dict(ncols=100, leave=False, file=sys.stdout)

    def __init__(self, config, fold, checkpoint=None):
        self.C = config
        self.fold = fold

        # Create Network
        if self.C.pose_model == "HRNet-W32":
            width = 32
        elif self.C.pose_model == "HRNet-W48":
            width = 48
        else:
            raise NotImplementedError()

        self.pose_model = networks.PoseHighResolutionNet(width)
        self.pose_model.load_state_dict(torch.load(f"networks/models/pose_hrnet_w{width}_384x288.pth"))

        final_layer = nn.Conv2d(width, 24, 1)
        with torch.no_grad():
            final_layer.weight[:17] = self.pose_model.final_layer.weight
            final_layer.bias[:17] = self.pose_model.final_layer.bias

            if self.C.model_additional_weight:
                # neck(17)은 left/right sholder(5, 6)과 nose(0)의 평균
                # left/right palm(18, 19)는 left/right wrist(9, 10)을 복사
                # spine2(20)은 left/right sholder(5, 6)과 left/right hip(11, 12)의 중앙
                # spine1(21)은 left/right hip(11, 12)을 각각 1/3 + left/right sholder(5, 6)을 각각 1/6
                # instep(22, 23)은 angle(15, 16)를 복사
                final_layer.weight[17] = self.pose_model.final_layer.weight[[0, 5, 6]].clone().mean(0)
                final_layer.bias[17] = self.pose_model.final_layer.bias[[0, 5, 6]].clone().mean(0)
                final_layer.weight[18] = self.pose_model.final_layer.weight[9].clone()
                final_layer.bias[18] = self.pose_model.final_layer.bias[9].clone()
                final_layer.weight[19] = self.pose_model.final_layer.weight[10].clone()
                final_layer.bias[19] = self.pose_model.final_layer.bias[10].clone()
                final_layer.weight[20] = self.pose_model.final_layer.weight[[5, 6, 11, 12]].clone().mean(0)
                final_layer.bias[20] = self.pose_model.final_layer.bias[[5, 6, 11, 12]].clone().mean(0)
                final_layer.weight[21] = torch.cat(
                    (
                        self.pose_model.final_layer.weight[[11, 12]].clone() * 1 / 3,
                        self.pose_model.final_layer.weight[[5, 6]].clone() * 6 / 1,
                    )
                ).mean(0)
                final_layer.bias[21] = torch.cat(
                    (
                        self.pose_model.final_layer.bias[[11, 12]].clone() * 1 / 3,
                        self.pose_model.final_layer.bias[[5, 6]].clone() * 6 / 1,
                    )
                ).mean(0)
                final_layer.weight[22] = self.pose_model.final_layer.weight[15].clone()
                final_layer.bias[22] = self.pose_model.final_layer.bias[15].clone()
                final_layer.weight[23] = self.pose_model.final_layer.weight[16].clone()
                final_layer.bias[23] = self.pose_model.final_layer.bias[16].clone()

            self.pose_model.final_layer = final_layer
        self.pose_model.cuda()

        # Criterion
        if self.C.train.loss_type == "ce":
            self.criterion = KeypointLoss().cuda()
        elif self.C.train.loss_type == "bce":
            self.criterion = KeypointBCELoss().cuda()
        elif self.C.train.loss_type == "mse":
            self.criterion = nn.MSELoss().cuda()
        elif self.C.train.loss_type == "mae":
            self.criterion = nn.L1Loss().cuda()
        elif self.C.train.loss_type == "awing":
            self.criterion = AWing().cuda()
        elif self.C.train.loss_type == "sigmae":
            self.criterion = SigmoidMAE().cuda()
        elif self.C.train.loss_type == "kldiv":
            self.criterion = SigmoidKLDivLoss().cuda()
        else:
            raise NotImplementedError()
        self.criterion_rmse = KeypointRMSE().cuda()

        # Optimizer
        if self.C.train.SAM:
            self.optimizer = utils.SAM(self.pose_model.parameters(), optim.AdamW, lr=self.C.train.lr)
        else:
            self.optimizer = optim.AdamW(self.pose_model.parameters(), lr=self.C.train.lr)

        self.epoch = 1
        self.best_loss = math.inf
        self.best_rmse = math.inf
        self.earlystop_cnt = 0

        # Dataset
        self.dl_train, self.dl_valid = get_pose_datasets(self.C, self.fold)

        # Load Checkpoint
        if checkpoint is not None and Path(checkpoint).exists():
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
                "model": self.pose_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": self.epoch,
                "best_loss": self.best_loss,
                "best_rmse": self.best_rmse,
                "earlystop_cnt": self.earlystop_cnt,
            },
            path,
        )

    def load(self, path):
        print("Load pretrained", path)
        ckpt = torch.load(path)
        self.pose_model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epoch = ckpt["epoch"] + 1
        self.best_loss = ckpt["best_loss"]
        self.best_rmse = ckpt["best_rmse"]
        self.earlystop_cnt = ckpt["earlystop_cnt"]

    def train_loop(self):
        self.pose_model.train()

        O = TrainOutput()
        with tqdm(total=len(self.dl_train.dataset), desc=f"Train {self.epoch:03d}", **self._tqdm_) as t:
            for files, imgs, target_heatmaps, ratios, offsets in self.dl_train:
                imgs_, target_heatmaps_ = imgs.cuda(non_blocking=True), target_heatmaps.cuda(non_blocking=True)

                # plus augment
                if self.C.train.plus_augment.do:
                    with torch.no_grad():
                        c = self.C.train.plus_augment
                        if c.downsample.do and random.random() <= c.downsample.p:
                            h, w = imgs_.shape[2:]
                            ratios[:, 0] = c.downsample.width / w * ratios[:, 0]
                            ratios[:, 1] = c.downsample.height / h * ratios[:, 1]
                            imgs_ = F.interpolate(imgs_, (c.downsample.height, c.downsample.width))
                            target_heatmaps_ = F.interpolate(
                                target_heatmaps_, (c.downsample.height // 4, c.downsample.width // 4)
                            )

                        if c.rotate.do and random.random() <= c.rotate.p:
                            k = 3 if random.random() < 0.5 else 1
                            ratios[:, 0], ratios[:, 1] = ratios[:, 1], ratios[:, 0]
                            imgs_ = torch.rot90(imgs_, k, dims=(2, 3))
                            target_heatmaps_ = torch.rot90(target_heatmaps_, k, dims=(2, 3))

                pred_heatmaps_ = self.pose_model(imgs_)
                loss = self.criterion(pred_heatmaps_, target_heatmaps_)
                rmse = self.criterion_rmse(pred_heatmaps_, target_heatmaps_, ratios.cuda(non_blocking=True))

                self.optimizer.zero_grad()
                loss.backward()
                if isinstance(self.optimizer, utils.SAM):
                    self.optimizer.first_step()
                    self.criterion(self.pose_model(imgs_), target_heatmaps_).backward()
                    self.optimizer.second_step()
                else:
                    self.optimizer.step()

                O.loss.update(loss.item(), len(files))
                O.rmse.update(rmse.item(), len(files))
                t.set_postfix_str(f"loss: {loss.item():.6f}, rmse: {rmse.item():.6f}", refresh=False)
                t.update(len(imgs))

        return O.freeze()

    @torch.no_grad()
    def valid_loop(self):
        self.pose_model.eval()

        O = TrainOutput()
        with tqdm(total=len(self.dl_valid.dataset), desc=f"Valid {self.epoch:03d}", **self._tqdm_) as t:
            for files, imgs, target_heatmaps, ratios, offsets in self.dl_valid:
                imgs_, target_heatmaps_ = imgs.cuda(non_blocking=True), target_heatmaps.cuda(non_blocking=True)
                pred_heatmaps_ = self.pose_model(imgs_)
                loss = self.criterion(pred_heatmaps_, target_heatmaps_)
                rmse = self.criterion_rmse(pred_heatmaps_, target_heatmaps_, ratios.cuda(non_blocking=True))

                O.loss.update(loss.item(), len(files))
                O.rmse.update(rmse.item(), len(files))
                t.set_postfix_str(f"loss: {loss.item():.6f}, rmse: {rmse.item():.6f}", refresh=False)
                t.update(len(imgs))

        return O.freeze()

    @torch.no_grad()
    def callback(self, to: TrainOutput, vo: TrainOutput):
        self.C.log.info(
            f"Epoch: {self.epoch:03d}/{self.C.train.max_epochs},",
            f"loss: {to.loss:.6f};{vo.loss:.6f},",
            f"rmse {to.rmse:.6f};{vo.rmse:.6f}",
        )
        self.C.log.flush()

        if isinstance(self.scheduler, utils.CosineAnnealingWarmUpRestarts):
            self.scheduler.step()
        elif isinstance(self.scheduler, lr_scheduler.CosineAnnealingWarmRestarts):
            self.scheduler.step()
        elif isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(vo.loss)

        if self.best_loss > vo.loss or self.best_rmse > vo.rmse:
            if self.best_loss > vo.loss:
                self.best_loss = vo.loss
            else:
                self.best_rmse = vo.rmse

            self.earlystop_cnt = 0
            self.save(self.C.result_dir / f"{self.C.uid}_{self.fold}.pth")
        else:
            self.earlystop_cnt += 1

    def fit(self):
        for self.epoch in range(self.epoch, self.C.train.max_epochs + 1):
            if self.C.train.finetune.do:
                if self.epoch <= self.C.train.finetune.step1_epochs:
                    if self.pose_model.finetune_step != 1:
                        self.C.log.info("Finetune step 1")
                        self.pose_model.freeze_step1()
                elif self.epoch <= self.C.train.finetune.step2_epochs:
                    if self.pose_model.finetune_step != 2:
                        self.C.log.info("Finetune step 2")
                        self.pose_model.freeze_step2()
                else:
                    if self.pose_model.finetune_step != 3:
                        self.C.log.info("Finetune step 3")
                        self.pose_model.freeze_step3()

            to = self.train_loop()
            vo = self.valid_loop()
            self.callback(to, vo)

            if self.earlystop_cnt > 10:
                self.C.log.info(f"Stop training at epoch", self.epoch)
                break


def main():
    args = argparse.ArgumentParser()
    args.add_argument("config", type=str)
    args = args.parse_args(sys.argv[1:])

    with open(args.config, "r") as f:
        C = EasyDict(yaml.load(f, yaml.FullLoader))

    for fold, checkpoint in zip(C.train.folds, C.train.checkpoints):
        with open(args.config, "r") as f:
            C = EasyDict(yaml.load(f, yaml.FullLoader))
            Path(C.result_dir).mkdir(parents=True, exist_ok=True)

            if C.dataset.num_cpus < 0:
                C.dataset.num_cpus = multiprocessing.cpu_count()
            C.uid = f"{C.pose_model}-{C.train.loss_type}-{C.dataset.input_width}x{C.dataset.input_height}"
            C.uid += "-plus_augment" if C.train.plus_augment.do else ""
            C.uid += "-sam" if C.train.SAM else ""
            C.uid += "-maw" if C.model_additional_weight else ""
            C.uid += f"-rr{C.dataset.ratio_limit:.1f}"
            C.uid += f"-{C.train.scheduler.type}"
            C.uid += f"-{C.comment}" if C.comment is not None else ""

            # log = utils.CustomLogger(Path(C.result_dir) / f"{C.uid}_{''.join(map(str, C.train.folds))}.log", "a")
            log = utils.CustomLogger(Path(C.result_dir) / f"{C.uid}_{fold}.log", "a")
            log.file.write("\r\n\r\n")
            log.info("\r\n" + pformat(C))
            log.flush()

            C.log = log
            C.result_dir = Path(C.result_dir)
            C.dataset.train_dir = Path(C.dataset.train_dir)
            utils.seed_everything(C.seed, deterministic=False)

        C.log.info("Fold", fold, ", checkpoint", checkpoint)
        trainer = PoseTrainer(C, fold, checkpoint)
        trainer.fit()


if __name__ == "__main__":
    main()
