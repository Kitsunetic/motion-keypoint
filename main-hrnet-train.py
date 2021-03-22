import json
import logging
import math
import os
import random
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import albumentations as A
import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.model_selection import KFold
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import networks
import utils
from error_list import error_list


POSE_MODEL = "HRNet-W48"
RESULT_DIR = Path("results/hrnet-train")
DATA_DIR = Path("data/ori")

LR = 1e-4  # transfer learning이니깐 좀 작게 주는게 좋을 것 같아서 1e-4
BATCH_SIZE = 10
START_EPOCH = 1
SAM = False
FOLD = 1 if len(sys.argv) == 1 else int(sys.argv[1])
PADDING = 30
ADD_JOINT_LOSS = False

DEBUG = False
STEP1_EPOCHS = 5
STEP2_EPOCHS = 10
STEP3_EPOCHS = 100
if DEBUG:
    STEP1_EPOCHS = 2
    STEP2_EPOCHS = 5
    STEP3_EPOCHS = 10

INPUT_WIDTH = 576  # 288
INPUT_HEIGHT = 768  # 384

n = datetime.now()
UID = f"{n.year:04d}{n.month:02d}{n.day:02d}-{n.hour:02d}{n.minute:02d}{n.second:02d}"
SEED = 20210309

utils.seed_everything(SEED, deterministic=False)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
log = utils.CustomLogger(RESULT_DIR / f"log_{UID}.log", "a")
log.info("학습 시작")
log.info("RESULT_DIR:", RESULT_DIR)
log.info("DATA_DIR:", DATA_DIR)
log.info("POSE_MODEL:", POSE_MODEL)
log.info("UID:", UID)
log.info("SEED:", SEED)
log.info("LR:", LR)
log.info("BATCH_SIZE:", BATCH_SIZE)
log.info("START_EPOCH:", START_EPOCH)
log.info("SAM:", SAM)
log.info("FOLD:", FOLD)
log.info("PADDING:", PADDING)
log.info("ADD_JOINT_LOSS:", ADD_JOINT_LOSS)
log.info("DEBUG:", DEBUG)
log.info("STEP1_EPOCHS:", STEP1_EPOCHS)
log.info("STEP2_EPOCHS:", STEP2_EPOCHS)
log.info("STEP3_EPOCHS:", STEP3_EPOCHS)
log.info("INPUT_WIDTH:", INPUT_WIDTH)
log.info("INPUT_HEIGHT:", INPUT_HEIGHT)
log.flush()


class JointMSELoss(nn.Module):
    def __init__(self, use_target_weight=False):
        super(JointMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction="mean")
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight=None):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(heatmap_pred.mul(target_weight[:, idx]), heatmap_gt.mul(target_weight[:, idx]))
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class KeypointLoss(nn.Module):
    def __init__(self, joint=False, use_target_weight=False):
        super().__init__()
        self.criterion = nn.MSELoss(reduction="mean")
        self.joint = joint
        self.use_target_weight = use_target_weight

    def forward(self, x, y):
        x = x.flatten(2).flatten(0, 1)
        y = y.flatten(2).flatten(0, 1).argmax(1)
        loss1 = F.cross_entropy(x, y)

        if self.joint:
            loss2 = self.joint_mse_loss(x, y)
            return loss1 + loss2
        else:
            return loss1

    def joint_mse_loss(self, output, target, target_weight=None):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(heatmap_pred.mul(target_weight[:, idx]), heatmap_gt.mul(target_weight[:, idx]))
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class KeypointRMSE(nn.Module):
    @torch.no_grad()
    def forward(self, pred_heatmaps: torch.Tensor, real_heatmaps: torch.Tensor, ratios: torch.Tensor):
        W = pred_heatmaps.size(3)
        pred_positions = pred_heatmaps.flatten(2).argmax(2)
        real_positions = real_heatmaps.flatten(2).argmax(2)
        pred_positions = torch.stack((pred_positions // W, pred_positions % W), 2).type(torch.float32)
        real_positions = torch.stack((real_positions // W, real_positions % W), 2).type(torch.float32)
        # print(pred_positions.shape, real_positions.shape, ratios.shape)
        pred_positions *= 4 / ratios.unsqueeze(1)  # position: (B, 24, 2), ratio: (B, 2)
        real_positions *= 4 / ratios.unsqueeze(1)
        loss = (pred_positions - real_positions).square().mean().sqrt()

        return loss


class KeypointDataset(Dataset):
    def __init__(self, files, keypoints, augmentation=True, padding=30):
        super().__init__()
        self.files = files
        self.keypoints = keypoints
        self.padding = padding

        T = []
        # T.append(A.Crop(0, 28, 1920, 1080 - 28))  # 1920x1080 --> 1920x1024
        # T.append(A.Resize(512, 1024))
        if augmentation:
            T.append(A.ImageCompression())
            T.append(A.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, value=0, rotate_limit=0))
            T.append(utils.HorizontalFlipEx())
            T.append(A.RandomRotate90())
            T.append(A.IAASharpen())  # 이거 뭔지?
            T.append(A.Cutout())
            T_ = []
            T_.append(A.RandomBrightnessContrast())
            T_.append(A.RandomGamma())
            T_.append(A.RandomBrightness())
            T_.append(A.RandomContrast())
            T.append(A.OneOf(T_))
            T.append(A.GaussNoise())
            T.append(A.Blur())
        T.append(A.Normalize())
        T.append(ToTensorV2())

        self.transform = A.Compose(
            transforms=T,
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
            # TODO 영역을 벗어난 keypoint는 그 영역의 한도 값으로 설정해줄 것?
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = str(self.files[idx])
        image = imageio.imread(file)

        keypoint = self.keypoints[idx]
        box = utils.keypoint2box(keypoint, self.padding)
        box = np.expand_dims(box, 0)
        labels = np.array([0], dtype=np.int64)
        a = self.transform(image=image, labels=labels, bboxes=box, keypoints=keypoint)

        image = a["image"]
        bbox = list(map(int, a["bboxes"][0]))
        keypoint = torch.tensor(a["keypoints"], dtype=torch.float32)
        image, keypoint, heatmap, ratio = self._resize_image(image, bbox, keypoint)

        return file, image, keypoint, heatmap, ratio

    def _resize_image(self, image, bbox, keypoint):
        # efficientdet에서 찾은 범위만큼 이미지를 자름
        image = image[:, bbox[1] : bbox[3], bbox[0] : bbox[2]]

        # HRNet의 입력 이미지 크기로 resize
        ratio = torch.tensor((INPUT_WIDTH / image.shape[2], INPUT_HEIGHT / image.shape[1]), dtype=torch.float32)
        image = F.interpolate(image.unsqueeze(0), (INPUT_HEIGHT, INPUT_WIDTH))[0]

        # bbox만큼 빼줌
        keypoint[:, 0] -= bbox[0]
        keypoint[:, 1] -= bbox[1]

        # 이미지를 resize해준 비율만큼 곱해줌
        keypoint[:, 0] *= ratio[0]
        keypoint[:, 1] *= ratio[1]
        # TODO: 잘못된 keypoint가 있으면 고쳐줌

        # HRNet은 1/4로 resize된 출력이 나오므로 4로 나눠줌
        keypoint /= 4

        # keypoint를 heatmap으로 변환
        # TODO: 완전히 정답이 아니면 틀린 것과 같은 점수. 좀 부드럽게 만들 수는 없을지?
        # heatmap regression loss중에 soft~~~ 한 이름이 있던거같은데
        heatmap = utils.keypoints2heatmaps(keypoint, INPUT_HEIGHT // 4, INPUT_WIDTH // 4)

        return image, keypoint, heatmap, ratio


class TestKeypointDataset(Dataset):
    def __init__(self, files, offsets, ratios, augmentation=False):
        super().__init__()
        self.files = files
        self.offsets = offsets
        self.ratios = ratios

        T = []
        if augmentation:
            T.append(A.ImageCompression())
            T.append(A.ShiftScaleRotate())
            # T.append(A.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, value=0))
            # T.append(utils.HorizontalFlipEx())
            # T.append(A.RandomRotate90())
            T.append(A.IAASharpen())  # 이거 뭔지?
            T.append(A.Cutout())
            T_ = []
            T_.append(A.RandomBrightnessContrast())
            T_.append(A.RandomGamma())
            T_.append(A.RandomBrightness())
            T_.append(A.RandomContrast())
            T.append(A.OneOf(T_))
            T.append(A.GaussNoise())
            T.append(A.Blur())
        T.append(A.Normalize())
        T.append(ToTensorV2())

        self.transform = A.Compose(transforms=T)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = str(self.files[idx])
        offset = torch.tensor(self.offsets[idx], dtype=torch.float32)
        ratio = torch.tensor(self.ratios[idx], dtype=torch.float32)

        image = imageio.imread(file)
        a = self.transform(image=image)
        image = a["image"]

        return file, image, offset, ratio


class Trainer:
    def __init__(self, fold=1, width=32, checkpoint=None):
        # Create Network
        assert width in [32, 48]

        self.pose_model = networks.PoseHighResolutionNet(width)
        self.pose_model.load_state_dict(torch.load(f"networks/models/pose_hrnet_w{width}_384x288.pth"))

        final_layer = nn.Conv2d(width, 24, 1)
        with torch.no_grad():
            final_layer.weight[:17] = self.pose_model.final_layer.weight
            final_layer.bias[:17] = self.pose_model.final_layer.bias
            self.pose_model.final_layer = final_layer
        self.pose_model.cuda()

        # Criterion
        self.criterion = KeypointLoss().cuda()
        self.criterion_rmse = KeypointRMSE().cuda()

        # Optimizer
        if SAM:
            self.optimizer = utils.SAM(self.pose_model.parameters(), optim.AdamW, lr=LR)
        else:
            self.optimizer = optim.AdamW(self.pose_model.parameters(), lr=LR)

        # Scheduler
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=4, verbose=True)

        self.epoch = 1
        self.best_loss = math.inf
        self.earlystop_cnt = 0

        # Dataset
        total_imgs = np.array(sorted(list((DATA_DIR / "train_imgs").glob("*.jpg"))))
        df = pd.read_csv("data/ori/train_df.csv")
        total_keypoints = df.to_numpy()[:, 1:].astype(np.float32)
        total_keypoints = np.stack([total_keypoints[:, 0::2], total_keypoints[:, 1::2]], axis=2)

        # 오류가 있는 데이터는 학습에서 제외
        total_imgs_, total_keypoints_ = [], []
        for i in range(len(total_imgs)):
            if i not in error_list:
                total_imgs_.append(total_imgs[i])
                total_keypoints_.append(total_keypoints[i])
        total_imgs = np.array(total_imgs_)
        total_keypoints = np.array(total_keypoints_)

        test_imgs = np.array(sorted(list((DATA_DIR / "test_imgs").glob("*.jpg"))))
        with open("data/test_imgs_effdet/data.json", "r") as f:
            data = json.load(f)
            offsets = data["offset"]
            ratios = data["ratio"]

        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        indices = list(kf.split(total_imgs))

        # train dataset
        train_idx, valid_idx = indices[0]
        ds_train = KeypointDataset(
            total_imgs[train_idx],
            total_keypoints[train_idx],
            augmentation=True,
            padding=PADDING,
        )
        self.dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)

        # validation dataset
        ds_valid = KeypointDataset(
            total_imgs[valid_idx],
            total_keypoints[valid_idx],
            augmentation=False,
            padding=PADDING,
        )
        self.dl_valid = DataLoader(ds_valid, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)

        # test dataset
        ds_test = TestKeypointDataset(test_imgs, offsets, ratios, augmentation=False)
        self.dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)

    def save(self, path):
        torch.save(
            {
                "model": self.pose_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": self.epoch,
                "best_loss": self.best_loss,
                "earlystop_cnt": self.earlystop_cnt,
                "UID": self.UID,
            },
            path,
        )

    def load(self, path):
        print("Load pretrained", path)
        ckpt = torch.load(path)
        self.pose_model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epoch = ckpt["epoch"]
        self.best_loss = ckpt["best_loss"]
        self.earlystop_cnt = ckpt["earlystop_cnt"]

        if "UID" in ckpt:
            UID = ckpt["UID"]

    def close(self):
        self.logger.close()

    def train_loop(self):
        self.pose_model.train()

        meanloss, meanrmse = utils.AverageMeter(), utils.AverageMeter()
        with tqdm(
            total=len(self.dl_train.dataset),
            ncols=100,
            leave=False,
            file=sys.stdout,
            desc=f"Train {self.epoch:03d}",
        ) as t:
            for files, imgs, keypoints, target_heatmaps, ratios in self.dl_train:
                imgs_, target_heatmaps_ = imgs.cuda(), target_heatmaps.cuda()
                pred_heatmaps_ = self.pose_model(imgs_)
                loss = self.criterion(pred_heatmaps_, target_heatmaps_)
                rmse = self.criterion_rmse(pred_heatmaps_, target_heatmaps_, ratios.cuda())

                self.optimizer.zero_grad()
                loss.backward()
                if isinstance(self.optimizer, utils.SAM):
                    self.optimizer.first_step()
                    self.criterion(self.pose_model(imgs_), target_heatmaps_).backward()
                    self.optimizer.second_step()
                else:
                    self.optimizer.step()

                meanloss.update(loss.item())
                meanrmse.update(rmse.item())
                t.set_postfix_str(f"loss: {loss.item():.6f}, rmse: {rmse.item():.6f}", refresh=False)
                t.update(len(imgs))

        return meanloss(), meanrmse()

    @torch.no_grad()
    def valid_loop(self):
        self.pose_model.eval()

        meanloss, meanrmse = utils.AverageMeter(), utils.AverageMeter()
        with tqdm(
            total=len(self.dl_valid.dataset),
            ncols=100,
            leave=False,
            file=sys.stdout,
            desc=f"Valid {self.epoch:03d}",
        ) as t:
            for files, imgs, keypoints, target_heatmaps, ratios in self.dl_valid:
                imgs_, target_heatmaps_ = imgs.cuda(), target_heatmaps.cuda()
                pred_heatmaps_ = self.pose_model(imgs_)
                loss = self.criterion(pred_heatmaps_, target_heatmaps_)
                rmse = self.criterion_rmse(pred_heatmaps_, target_heatmaps_, ratios.cuda())

                meanloss.update(loss.item())
                meanrmse.update(rmse.item())
                t.set_postfix_str(f"loss: {loss.item():.6f}, rmse: {rmse.item():.6f}", refresh=False)
                t.update(len(imgs))

        return meanloss(), meanrmse()

    def finetune_step1(self):
        log.info("Finetune Step 1")
        self.pose_model.freeze_head()
        for self.epoch in range(self.epoch, STEP1_EPOCHS + 1):
            tloss, trmse = self.train_loop()
            vloss, vrmse = self.valid_loop()

            log.info(
                f"Epoch: {self.epoch:03d},",
                f"loss: {tloss:.6f};{vloss:.6f},",
                f"rmse {trmse:.6f};{vrmse:.6f}",
            )
            log.flush()

            if self.best_loss > vloss:
                self.best_loss = vloss
                self.save(RESULT_DIR / f"ckpt-{UID}_{FOLD}.pth")

    def finetune_step2(self):
        log.info("Finetune Step 2")
        self.pose_model.freeze_tail()
        for self.epoch in range(self.epoch, STEP2_EPOCHS + 1):
            tloss, trmse = self.train_loop()
            vloss, vrmse = self.valid_loop()

            log.info(
                f"Epoch: {self.epoch:03d},",
                f"loss: {tloss:.6f};{vloss:.6f},",
                f"rmse {trmse:.6f};{vrmse:.6f}",
            )
            log.flush()

            if self.best_loss > vloss:
                self.best_loss = vloss
                self.save(RESULT_DIR / f"ckpt-{UID}_{FOLD}.pth")

    def finetune_step3(self):
        log.info("Finetune Step 3")
        self.pose_model.unfreeze_all()
        for self.epoch in range(self.epoch, STEP3_EPOCHS + 1):
            tloss, trmse = self.train_loop()
            vloss, vrmse = self.valid_loop()

            log.info(
                f"Epoch: {self.epoch:03d},",
                f"loss: {tloss:.6f};{vloss:.6f},",
                f"rmse {trmse:.6f};{vrmse:.6f}",
            )
            log.flush()
            self.scheduler.step(vloss)

            if self.best_loss > vloss:
                self.best_loss = vloss
                self.earlystop_cnt = 0
                self.save(RESULT_DIR / f"ckpt-{UID}_{FOLD}.pth")
            elif self.earlystop_cnt >= 10:
                log.info(f"Stop training at epoch", self.epoch)
                break
            else:
                self.earlystop_cnt += 1

    def fit(self):
        self.finetune_step1()
        torch.cuda.empty_cache()
        self.load(RESULT_DIR / f"ckpt-{UID}_{FOLD}.pth")

        self.finetune_step2()
        torch.cuda.empty_cache()
        self.load(RESULT_DIR / f"ckpt-{UID}_{FOLD}.pth")

        self.finetune_step3()
        torch.cuda.empty_cache()
        self.load(RESULT_DIR / f"ckpt-{UID}_{FOLD}.pth")


def main():
    for fold in [1, 2, 3, 4, 5]:
        log.info("Fold", fold)
        trainer = Trainer(fold, width=48)

        trainer.fit()


if __name__ == "__main__":
    main()
