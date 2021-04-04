import argparse
import json
import math
import os
import shutil
import sys
from multiprocessing import cpu_count
from pathlib import Path

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
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data.dataloader import DataLoader
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
        if self.C.det_model.pretrained is not None:
            self.load(self.C.det_model.pretrained)

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
            self.save(Path(self.C.result_dir) / f"ckpt-{self.C.uid}_{self.fold}.pth")
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

        self.load(Path(self.C.result_dir) / f"ckpt-{self.C.uid}_{self.fold}.pth")

    def scale_inference_single_image(self, img, dw, dh, flip=False):
        _, h, w = img.shape
        img = img.unsqueeze(0)

        ratio_x = dw / w
        ratio_y = dh / h
        img = F.interpolate(img, (dh, dw))
        if flip:
            img = torch.flip(img, dims=[3])
        result = self.det_model(img)[0]

        class_mask = result["class_ids"] == 0
        rois = result["rois"][class_mask]
        scores = result["scores"][class_mask]

        if len(rois) > 0:
            j = np.argmax(scores)
            roi = rois[j]
            score = scores[j]

            if flip:
                roi[0::2] = (dw - roi[0::2]) / ratio_x
                roi[1::2] /= ratio_y
                temp = roi[0].copy()
                roi[0] = roi[2]
                roi[2] = temp
            else:
                roi[0::2] /= ratio_x
                roi[1::2] /= ratio_y

            return roi, score
        else:
            return None, None

    def multiscale_inference(self, imgs):
        """사이즈/LR에 대해 bbox 좌표를 구하고, voting이 아니라 좌표값에 대한 가중평균을 구함"""
        rectified_rois = []

        for img in imgs:
            rois = []
            scores = []
            for dw, dh in self.C.inference.multiscale_test.sizes:
                roi, score = self.scale_inference_single_image(img, dw, dh, flip=False)
                if roi is not None:
                    rois.append(roi)
                    scores.append(score)

                if self.C.inference.flip_test.horizontal:
                    roi, score = self.scale_inference_single_image(img, dw, dh, flip=True)
                    if roi is not None:
                        rois.append(roi)
                        scores.append(score)

            # print(len(rois) )

            rois = np.stack(rois)
            scores = np.stack(scores)
            new_roi = np.average(rois, axis=0, weights=scores)
            rectified_rois.append(new_roi)

        return np.stack(rectified_rois)

    @torch.no_grad()
    def test_loop(self, file_out_dir):
        self.det_model.eval()
        file_out_dir = Path(file_out_dir)
        file_out_dir.mkdir(parents=True, exist_ok=True)

        inf = self.C.inference

        with tqdm(total=len(self.dl_test.dataset), ncols=100, file=sys.stdout) as t:
            for files, imgs in self.dl_test:
                imgs_ = imgs.cuda(non_blocking=True)

                # multi-scale test
                pred_bboxes = self.multiscale_inference(imgs_)

                for file, pred_bbox in zip(files, pred_bboxes):
                    file = Path(file)
                    t.set_postfix_str(file.name)

                    ud_bbox = pred_bbox.copy()
                    ud_bbox[0::2] = ud_bbox[0::2] / self.C.input_width * (1920 - self.C.crop[0] * 2)
                    ud_bbox[1::2] = ud_bbox[1::2] / self.C.input_height * (1080 - self.C.crop[1] * 2)
                    ud_bbox[0::2] += self.C.crop[0]
                    ud_bbox[1::2] += self.C.crop[1]
                    int_bbox = ud_bbox.astype(np.int64)

                    img_ori = imageio.imread(file)
                    clip = img_ori[int_bbox[1] : int_bbox[3], int_bbox[0] : int_bbox[2]]
                    imageio.imwrite(file_out_dir / file.name, clip)

                    t.update()


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
        C.uid += f"-sam" if C.train.SAM else ""
        C.uid += f"-{C.dataset.input_width}x{C.dataset.input_height}"
        C.uid += f"-pad{C.dataset.padding}"
        C.uid += f"-{C.comment}" if C.comment is not None else ""
        C.uid += f"_{C.train.fold}"

        trainer = DetTrainer(C, fold, checkpoint)
        trainer.fit()


if __name__ == "__main__":
    main()
