import math
import os
import random
import shutil
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from io import TextIOWrapper
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
import torch.optim as optim
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import KFold
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from tqdm import tqdm

import utils
import networks


RESULT_DIR = Path("results/HRNet학습")

LR = 1e-4  # transfer learning이니깐 좀 작게 주는게 좋을 것 같아서 1e-4
BATCH_SIZE = 9
START_EPOCH = 1
SAM = False
FOLDS = [1, 2, 3, 4, 5]
HRNET_WIDTH = 48
USE_L1 = False

n = datetime.now()
UID = f"{n.year:04d}{n.month:02d}{n.day:02d}-{n.hour:02d}{n.minute:02d}{n.second:02d}"
SEED = 20210309


class ImageDataset(Dataset):
    def __init__(self, files, offsets, keypoints=None):
        super().__init__()
        self.files = files
        self.offsets = offsets
        self.keypoints = keypoints

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        img = imageio.imread(f)
        H, W, _ = img.shape
        # TODO 가로로 긴 영상이면 가로 길이가 768이 되도록 만들기
        ratio = torch.tensor([576 / W, 768 / H], dtype=torch.float32)
        img = cv2.resize(img, (576, 768))
        x = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        offset = torch.tensor(self.offsets[idx]["boxes"][:2], dtype=torch.int64)

        if self.keypoints is not None:
            keypoint = torch.tensor(self.keypoints[idx], dtype=torch.float32)
            keypoint[:, 0] = (keypoint[:, 0] - offset[0]) * ratio[0] / 4
            keypoint[:, 1] = (keypoint[:, 1] - offset[1]) * ratio[1] / 4
            keypoint = keypoint.type(torch.int64)
            # TODO: 나중에 augmentation 추가

            """# 좌표값 keypoint를 24차원 평면으로 변환
            y = torch.zeros(24, 768 // 4, 576 // 4, dtype=torch.int64)
            for i in range(24):
                y[i, keypoint[i, 1] // 4, keypoint[i, 0] // 4] = 1"""
            # 좌표값 keypoint를 1차원 벡터의 위치 값으로 변환
            y = keypoint[:, 0] + keypoint[:, 1] * (576 // 4)

            return f.name, x, offset, ratio, y
        return f.name, x, offset, ratio


class KeypointLoss(nn.Module):
    def forward(self, x, y):
        x = x.flatten(2).flatten(0, 1)
        y = y.flatten(0, 1)
        return F.cross_entropy(x, y)


class KeypointRMSE(nn.Module):
    @torch.no_grad()
    def forward(self, x, y):
        W = x.size(3)
        xp = x.flatten(2).argmax(2)
        xx, xy = xp % W, xp // W
        yx, yy = y % W, y // W
        return 4 * ((xx - yx) ** 2 + (xy - yy) ** 2).type(torch.float32).mean().sqrt()


def train_loop(dl: DataLoader, model, epoch, criterion, criterion_rmse, optimizer):
    torch.cuda.empty_cache()
    model.train()

    meanloss = utils.AverageMeter()
    meanrmse = utils.AverageMeter()
    results = {"image": [], "loss": [], "rmse": []}
    with tqdm(total=len(dl.dataset), ncols=100, leave=False, file=sys.stdout, desc=f"Train[{epoch:03d}]") as t:
        for f, x, offset, ratio, y in dl:
            x_ = x.cuda()
            y_ = y.cuda()
            p_ = model(x_)
            loss = criterion(p_, y_)
            rmse = criterion_rmse(p_, y_)

            optimizer.zero_grad()
            loss.backward()
            if isinstance(optimizer, utils.SAM):
                optimizer.first_step()
                loss = criterion(model(x_), y_).backward()
                optimizer.second_step()
            else:
                optimizer.step()

            meanloss.update(loss.item())
            meanrmse.update(rmse.item())
            results["image"].append(f)
            results["loss"].append(loss.item())
            results["rmse"].append(rmse.item())
            t.set_postfix_str(f"loss: {loss.item():.6f}, rmse: {rmse.item():.6f}", refresh=False)
            t.update(len(x))

    return meanloss(), meanrmse(), results


@torch.no_grad()
def valid_loop(dl: DataLoader, model, epoch, criterion, criterion_rmse, optimizer):
    torch.cuda.empty_cache()
    model.eval()

    meanloss = utils.AverageMeter()
    meanrmse = utils.AverageMeter()
    results = {"image": [], "loss": [], "rmse": []}
    with tqdm(total=len(dl.dataset), ncols=100, leave=False, file=sys.stdout, desc=f"Valid[{epoch:03d}]") as t:
        for f, x, offset, ratio, y in dl:
            x_ = x.cuda()
            y_ = y.cuda()
            p_ = model(x_)
            loss = criterion(p_, y_)
            rmse = criterion_rmse(p_, y_)

            meanloss.update(loss.item())
            meanrmse.update(rmse.item())
            results["image"].append(f)
            results["loss"].append(loss.item())
            results["rmse"].append(rmse.item())
            t.set_postfix_str(f"loss: {loss.item():.6f}, rmse: {rmse.item():.6f}", refresh=False)
            t.update(len(x))

    return meanloss(), meanrmse(), results


def main(fold):
    utils.seed_everything(SEED, deterministic=False)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    log = utils.CustomLogger(RESULT_DIR / f"log_{UID}.log", "a")
    log.info("학습 시작")
    log.info("UID:", UID)
    log.info("SEED:", SEED)
    log.info("LR:", LR)
    log.info("BATCH_SIZE:", BATCH_SIZE)
    log.info("START_EPOCH:", START_EPOCH)
    log.info("SAM:", SAM)
    log.info("FOLD:", fold)
    log.info("HRNET_WIDTH:", HRNET_WIDTH)
    log.info("USE_L1:", USE_L1)

    train_imgs = np.array(sorted(list(Path("data/box2/train_imgs/").glob("*.jpg"))))
    test_imgs = np.array(sorted(list(Path("data/box2/test_imgs/").glob("*.jpg"))))
    keypoints = pd.read_csv("data/ori/train_df.csv").to_numpy()[:, 1:].astype(np.float32)
    keypoints = np.stack([keypoints[:, 0::2], keypoints[:, 1::2]], axis=2)
    with open("data/box2/offset.json", "r") as f:
        offsets = json.load(f)

    ds_train_total = ImageDataset(train_imgs, offsets["train"], keypoints)
    ds_test = ImageDataset(test_imgs, offsets["test"])
    dl_kwargs = dict(batch_size=BATCH_SIZE, num_workers=4)

    model = networks.PoseHighResolutionNet(width=HRNET_WIDTH)
    model.load_state_dict(torch.load(f"networks/models/pose_hrnet_w{HRNET_WIDTH}_384x288.pth"))
    final_layer = nn.Conv2d(HRNET_WIDTH, 24, 1)
    with torch.no_grad():
        final_layer.weight[:17] = model.final_layer.weight
        final_layer.bias[:17] = model.final_layer.bias
    model.final_layer = final_layer
    model = model.cuda()

    if USE_L1:
        criterion = nn.L1Loss().cuda()
    else:
        criterion = KeypointLoss().cuda()
    criterion_rmse = KeypointRMSE().cuda()

    if SAM:
        optimizer = utils.SAM(model.parameters(), optim.AdamW, lr=LR)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=LR)

    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=4, verbose=True)

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, valid_idx = list(kf.split(ds_train_total))[fold - 1]

    ds_train = Subset(ds_train_total, train_idx)
    ds_valid = Subset(ds_train_total, valid_idx)
    dl_train = DataLoader(ds_train, **dl_kwargs, shuffle=True)
    dl_valid = DataLoader(ds_valid, **dl_kwargs, shuffle=False)

    best_loss = math.inf
    early_stop_cnt = 0

    for epoch in range(START_EPOCH, 999):
        tloss, trmse, tres = train_loop(dl_train, model, epoch, criterion, criterion_rmse, optimizer)
        vloss, vrmse, vres = valid_loop(dl_valid, model, epoch, criterion, criterion_rmse, optimizer)

        # Logging
        log.info(f"Epoch: {epoch:03d}, loss: {tloss:.6f} ; {vloss:.6f}, rmse {trmse:.6f} ; {vrmse:.6f}")
        scheduler.step(vloss)

        # Earlystop
        if vloss < best_loss:
            best_loss = vloss
            early_stop_cnt = 0

            with open(RESULT_DIR / f"loss-{UID}.json", "w") as f:
                json.dump({"train": tres, "valid": vres}, f)

            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                },
                RESULT_DIR / f"ckpt-{UID}_{fold}.pth",
            )
        elif early_stop_cnt >= 10:
            log.info(f"Stop training at epoch {epoch} of fold {fold}.")
            break
        else:
            early_stop_cnt += 1


if __name__ == "__main__":
    for fold in FOLDS:
        main(fold)
