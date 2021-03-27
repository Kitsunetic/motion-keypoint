import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm

import networks
import options
import utils
from datasets import get_det_dataset


class DetTrainer:
    def __init__(self, config, fold):
        self.config = config
        self.fold = fold

        self.det_model = networks.EfficientDetFinetune(config.det_model, pretrained=True, finetune=config.finetune.do)
        self.det_model.cuda()

        # Optimizer
        if config.SAM:
            self.optimizer = utils.SAM(self.det_model.parameters(), optim.AdamW, lr=config.lr)
        else:
            self.optimizer = optim.AdamW(self.det_model.parameters(), lr=config.lr)

        self.epoch = config.start_epoch
        self.best_loss = math.inf
        self.earlystop_cnt = 0

        # Dataset
        self.dl_train, self.dl_valid, self.dl_test = get_det_dataset(config, self.fold)

        # Load Checkpoint
        if config.pretrained is not None:
            self.load(config.pretrained)

        # Scheduler
        self.scheduler = options.get_scheduler(self.config, self.optimizer, self.epoch - 2)

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

        meanloss = utils.AverageMeter()
        with tqdm(
            total=len(self.dl_train.dataset),
            ncols=100,
            leave=False,
            file=sys.stdout,
            desc=f"Train {self.epoch:03d}",
        ) as t:
            for files, imgs, annots in self.dl_train:
                imgs_, annots_ = imgs.cuda(non_blocking=True), annots.cuda(non_blocking=True)
                loss = self.det_model(imgs_, annots_)

                self.optimizer.zero_grad()
                loss.backward()
                if isinstance(self.optimizer, utils.SAM):
                    self.optimizer.first_step()
                    self.det_model(imgs_, annots_).backward()
                    self.optimizer.second_step()
                else:
                    self.optimizer.step()

                meanloss.update(loss.item())
                t.set_postfix_str(f"loss: {loss.item():.6f}", refresh=False)
                t.update(len(imgs))

        return meanloss()

    @torch.no_grad()
    def valid_loop(self):
        self.det_model.eval()

        meanloss = utils.AverageMeter()
        with tqdm(
            total=len(self.dl_valid.dataset),
            ncols=100,
            leave=False,
            file=sys.stdout,
            desc=f"Valid {self.epoch:03d}",
        ) as t:
            for files, imgs, annots in self.dl_valid:
                imgs_, annots_ = imgs.cuda(non_blocking=True), annots.cuda(non_blocking=True)
                loss = self.det_model(imgs_, annots_)

                meanloss.update(loss.item())
                t.set_postfix_str(f"loss: {loss.item():.6f}", refresh=False)
                t.update(len(imgs))

        return meanloss()

    @torch.no_grad()
    def test_loop(self, file_out_dir):
        self.det_model.eval()
        file_out_dir = Path(file_out_dir)
        file_out_dir.mkdir(parents=True, exist_ok=True)

        with tqdm(total=len(self.dl_test.dataset), ncols=100, file=sys.stdout) as t:
            for files, imgs in self.dl_test:
                imgs_ = imgs.cuda(non_blocking=True)
                pred_bboxes = self.det_model(imgs_)

                for file, img, pred_bbox in zip(files, imgs, pred_bboxes):
                    pred_bbox = pred_bbox["rois"][0]
                    ud_bbox = pred_bbox.copy()
                    ud_bbox[0::2] = ud_bbox[0::2] / self.config.input_width * 1920 + self.config.crop[0]
                    ud_bbox[1::2] = ud_bbox[1::2] / self.config.input_height * 1080 + self.config.crop[1]
                    int_bbox = ud_bbox.astype(np.int64)

                    file = Path(file)
                    t.set_postfix_str(file.name)

                    img_ori = imageio.imread(file)
                    clip = img_ori[int_bbox[1] : int_bbox[3], int_bbox[0] : int_bbox[2]]
                    imageio.imwrite(file_out_dir / file.name, clip)

                    t.update()

    @torch.no_grad()
    def callback(self, tloss, vloss):
        self.config.log.info(
            f"Epoch: {self.epoch:03d},",
            f"loss: {tloss:.6f};{vloss:.6f},",
        )
        self.config.log.flush()

        if isinstance(self.scheduler, lr_scheduler.CosineAnnealingWarmRestarts):
            self.scheduler.step()
        elif isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(vloss)

        if self.best_loss > vloss:
            self.best_loss = vloss
            self.earlystop_cnt = 0
            self.save(self.config.result_dir / f"ckpt-{self.config.uid}_{self.fold}.pth")
        else:
            self.earlystop_cnt += 1

    def fit(self):
        for self.epoch in range(self.epoch, self.config.final_epoch + 1):
            if self.config.finetune.do:
                if self.epoch <= self.config.finetune.step1_epochs:
                    self.det_model.unfreeze_tail()
                elif self.epoch <= self.config.finetune.step2_epochs:
                    self.det_model.unfreeze_head()
                else:
                    self.det_model.unfreeze()

            tloss = self.train_loop()
            vloss = self.valid_loop()
            self.callback(tloss, vloss)

            if self.earlystop_cnt > self.config.earlystop_patience:
                self.config.log.info(f"Stop training at epoch", self.epoch)
                break

        self.load(self.config.result_dir / f"ckpt-{self.config.uid}_{self.fold}.pth")


def main():
    args = argparse.ArgumentParser()
    args.add_argument("config_file", type=str)
    args = args.parse_args(sys.argv[1:])

    config = options.load_config(args.config_file)
    trainer = DetTrainer(config, 1)
    trainer.fit()

    # validation exmaple 이미지 저장
    trainer.test_loop(config.result_dir / "example" / f"valid")


if __name__ == "__main__":
    main()
