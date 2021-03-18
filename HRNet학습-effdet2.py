import json
import math
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

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
from sklearn.model_selection import KFold
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import networks
import utils
from error_list import error_list

""" 
=================================================================================
Hyper Parameters
================================================================================= 
"""


RESULT_DIR = Path("results/HRNet학습-effdet2")

MODEL = "HRNet48"
LR = 1e-4  # transfer learning이니깐 좀 작게 주는게 좋을 것 같아서 1e-4
BATCH_SIZE = 10
START_EPOCH = 1
SAM = True
FOLDS = [1, 2, 3, 4, 5]
LOSS_TYPE = "BCE"  # CE, BCE, L1
AUG_HORIZONTAL_FLIP = True
AUG_SHIFT = True

n = datetime.now()
UID = f"{n.year:04d}{n.month:02d}{n.day:02d}-{n.hour:02d}{n.minute:02d}{n.second:02d}"
SEED = 20210309

NORMALIZE = False
MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).reshape(3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).reshape(3, 1, 1)

utils.seed_everything(SEED, deterministic=False)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
log = utils.CustomLogger(RESULT_DIR / f"log_{UID}.log", "a")
log.info("학습 시작")
log.info("UID:", UID)
log.info("SEED:", SEED)
log.info("MODEL:", MODEL)
log.info("LR:", LR)
log.info("BATCH_SIZE:", BATCH_SIZE)
log.info("START_EPOCH:", START_EPOCH)
log.info("SAM:", SAM)
log.info("FOLD:", FOLDS)
log.info("LOSS_TYPE:", LOSS_TYPE)
log.info("AUG_HORIZONTAL_FLIP:", AUG_HORIZONTAL_FLIP)
log.info("AUG_SHIFT:", AUG_SHIFT)
log.info("NORMALIZE:", NORMALIZE)
log.info("NORMALIZE_MEAN:", MEAN.squeeze().tolist())
log.info("NORMALIZE_STD:", STD.squeeze().tolist())


""" 
=================================================================================
Dataset
================================================================================= 
"""


df = pd.read_csv("data/ori/train_df.csv")
c2i = {c: i for i, c in enumerate(df.columns[1:])}
swap_columns = []
for i, c in enumerate(df.columns[1:]):
    if c.startswith("left_") and c.endswith("_x"):
        swap_columns.append((i // 2, c2i["right_" + c[5:]] // 2))


def horizontal_flip(x, keypoints, p=0.5):
    if random.random() > p:
        return x, keypoints

    dx = torch.flip(x, dims=(2,))
    maxw = x.size(2) // 4
    keypoints[:, 0] = maxw - keypoints[:, 0]
    for a, b in swap_columns:
        temp = keypoints[a].clone()
        keypoints[a] = keypoints[b].clone()
        keypoints[b] = temp

    return dx, keypoints


def random_shift(x, keypoints, distance=20, p=0.5):
    if random.random() > 1:
        return x, keypoints

    _, H, W = x.shape

    # distance = min(keypoints[:, 0].min(), W - keypoints[:, 0].max(), keypoints[:, 1].min(), H - keypoints[:, 1].max())
    # print(distance)

    # yp = random.randint(-distance, distance)
    # xp = random.randint(-distance, distance)
    xp = random.randint(-8, 8)
    yp = random.randint(-24, 24)
    keypoints[:, 0] += xp // 4
    keypoints[:, 1] += yp // 4

    dx = torch.zeros_like(x)
    dxl = xp if xp >= 0 else 0
    dxr = W if xp >= 0 else W + xp
    dxt = yp if yp >= 0 else 0
    dxb = H if yp >= 0 else H + yp
    xl = 0 if xp >= 0 else -xp
    xr = W - xp if xp >= 0 else W
    xt = 0 if yp >= 0 else -yp
    xb = H - yp if yp >= 0 else H
    dx[..., dxt:dxb, dxl:dxr] = x[..., xt:xb, xl:xr]

    return dx, keypoints


class ImageDataset(Dataset):
    def __init__(self, imdir, offsets, keypoints=None, augmentation=True):
        super().__init__()
        self.imdir = Path(imdir)
        self.offsets = offsets
        self.keypoints = keypoints
        self.augmentation = augmentation

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        f = self.imdir / self.offsets[idx]["image"]
        x = imageio.imread(f)
        x = torch.tensor(x, dtype=torch.float32).permute(2, 0, 1) / 255.0

        # Standardization. HRNet도 standardization 하고 있음.
        # Pretrained weight를 제대로 쓰고 싶다면 standardization은 필수
        if NORMALIZE:
            x = (x - MEAN) / STD

        # box값으로 전체 이미지에 대한 keypoint와 국소 keypoint로 서로 왕래할 수 있음
        ratio = torch.tensor(self.offsets[idx]["ratio"], dtype=torch.float32)
        offset = torch.tensor(self.offsets[idx]["boxes"][:2], dtype=torch.int64)

        if self.keypoints is not None:
            keypoint = torch.tensor(self.keypoints[idx], dtype=torch.float32)
            keypoint[:, 0] = (keypoint[:, 0] - offset[0]) * ratio / 4
            keypoint[:, 1] = (keypoint[:, 1] - offset[1]) * ratio / 4
            keypoint = keypoint.type(torch.int64)

            # Augmentation
            if self.augmentation:
                if AUG_HORIZONTAL_FLIP:
                    x, keypoint = horizontal_flip(x, keypoint, p=0.5)
                if AUG_SHIFT:
                    x, keypoint = random_shift(x, keypoint, p=0.5)

            if LOSS_TYPE in ["BCE", "L1"]:
                # 좌표값 keypoint를 24차원 평면으로 변환 --> BCE Loss or L1 Loss
                y = utils.keypoints2channels(keypoint)
            elif LOSS_TYPE == "CE":
                # 좌표값 keypoint를 1차원 벡터의 위치 값으로 변환 --> 각 keypoint마다 cross entropy 이후 reduction.
                # TODO cross entropy는 가로세로 위치에 대한 영향은 신경쓰지 않음. 이 부분 어찌 향상시킬 방법 없을까?
                y = keypoint[:, 0] + keypoint[:, 1] * (576 // 4)
            else:
                raise NotImplementedError()

            return f.name, x, offset, ratio, y
        return f.name, x, offset, ratio


train_imdir = Path("data/box_effdet2/train_imgs/")
test_imdir = Path("data/box_effdet2/test_imgs/")
df = pd.read_csv("data/box_effdet2/train_df.csv")
keypoints = df.to_numpy()[:, 1:].astype(np.float32)
keypoints = np.stack([keypoints[:, 0::2], keypoints[:, 1::2]], axis=2)
with open("data/box_effdet2/offset.json", "r") as f:
    offsets = json.load(f)

# 잘못 매칭된 키포인드들 제거
offsets_ = []
keypoints_ = []
for i, (offset, keypoint) in enumerate(zip(offsets["train"], keypoints)):
    if i not in error_list:
        offsets_.append(offset)
        keypoints_.append(keypoint)

offsets["train"] = offsets_
keypoints = np.stack(keypoints_)

ds_train_total = ImageDataset(train_imdir, offsets["train"], keypoints, augmentation=False)
ds_test = ImageDataset(test_imdir, offsets["test"])
dl_kwargs = dict(batch_size=BATCH_SIZE, num_workers=4)


""" 
=================================================================================
Model Ready
================================================================================= 
"""


class KeypointLoss(nn.Module):
    if LOSS_TYPE == "CE":

        def forward(self, x: torch.Tensor, y: torch.Tensor):
            x = x.flatten(2).flatten(0, 1)
            y = y.flatten(0, 1)
            return F.cross_entropy(x, y)

    elif LOSS_TYPE == "BCE":

        def forward(self, x: torch.Tensor, y: torch.Tensor):
            return F.binary_cross_entropy(x.sigmoid(), y)

    elif LOSS_TYPE == "L1":

        def forward(self, x: torch.Tensor, y: torch.Tensor):
            raise NotImplementedError()


class KeypointRMSE(nn.Module):
    if LOSS_TYPE == "CE":

        @torch.no_grad()
        def forward(self, x: torch.Tensor, y: torch.Tensor, ratio: torch.Tensor):
            W = x.size(3)
            xp = x.flatten(2).argmax(2)
            xx, xy = xp % W, xp // W
            yx, yy = y % W, y // W

            loss = (4 * ((xx - yx) ** 2 + (xy - yy) ** 2) * (ratio ** 2)).type(torch.float32)
            return loss.mean().sqrt()

    elif LOSS_TYPE == "BCE":

        @torch.no_grad()
        def forward(self, x: torch.Tensor, y: torch.Tensor, ratio: torch.Tensor):
            pos = x.flatten(2).argmax(2).type(torch.int64)  # N, 24
            pos_y = pos // 144
            pos_x = pos % 144
            xk = torch.stack([pos_x, pos_y], 2).type(torch.float32)
            xk /= ratio.unsqueeze(-1).unsqueeze(-1)

            pos = y.flatten(2).argmax(2).type(torch.int64)  # N, 24
            pos_y = pos // 144
            pos_x = pos % 144
            yk = torch.stack([pos_x, pos_y], 2).type(torch.float32)
            yk /= ratio.unsqueeze(-1).unsqueeze(-1)

            return 4 * (xk - yk).square().mean().sqrt()


def get_model():
    if MODEL == "HRNet48":
        model = networks.PoseHighResolutionNet(width=48)
        model.load_state_dict(torch.load(f"networks/models/pose_hrnet_w48_384x288.pth"))
        final_layer = nn.Conv2d(48, 24, 1)
        with torch.no_grad():
            final_layer.weight[:17] = model.final_layer.weight
            final_layer.bias[:17] = model.final_layer.bias
        model.final_layer = final_layer
        return model
    if MODEL == "HRNet32":
        model = networks.PoseHighResolutionNet(width=32)
        model.load_state_dict(torch.load(f"networks/models/pose_hrnet_w32_384x288.pth"))
        final_layer = nn.Conv2d(32, 24, 1)
        with torch.no_grad():
            final_layer.weight[:17] = model.final_layer.weight
            final_layer.bias[:17] = model.final_layer.bias
        model.final_layer = final_layer
        return model


class SupplyBean:
    def __init__(self):
        self.model = get_model().cuda()
        self.criterion = KeypointLoss().cuda()
        self.criterion_rmse = KeypointRMSE().cuda()
        if SAM:
            self.optimizer = utils.SAM(self.model.parameters(), optim.AdamW, lr=LR)
        else:
            self.optimizer = optim.AdamW(self.model.parameters(), lr=LR)
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=4, verbose=True)

        self.epoch = 1
        self.dl_train = dl_train
        self.dl_valid = dl_valid


""" 
=================================================================================
Training Ready
================================================================================= 
"""


def train_loop(B: SupplyBean, dl: DataLoader):
    B.model.train()

    meanloss = utils.AverageMeter()
    meanrmse = utils.AverageMeter()
    results = {"image": [], "loss": [], "rmse": []}
    with tqdm(total=len(dl.dataset), ncols=100, leave=False, file=sys.stdout, desc=f"Train[{B.epoch:03d}]") as t:
        for f, x, offset, ratio, y in dl:
            x_ = x.cuda()
            y_ = y.cuda()
            p_ = B.model(x_)
            loss = B.criterion(p_, y_)
            rmse = B.criterion_rmse(p_, y_, ratio.cuda())

            B.optimizer.zero_grad()
            loss.backward()
            if isinstance(B.optimizer, utils.SAM):
                B.optimizer.first_step()
                B.criterion(B.model(x_), y_).backward()
                B.optimizer.second_step()
            else:
                B.optimizer.step()

            meanloss.update(loss.item())
            meanrmse.update(rmse.item())
            results["image"].append(f)
            results["loss"].append(loss.item())
            results["rmse"].append(rmse.item())
            t.set_postfix_str(f"loss: {loss.item():.6f}, rmse: {rmse.item():.6f}", refresh=False)
            t.update(len(x))

    return meanloss(), meanrmse(), results


@torch.no_grad()
def valid_loop(B: SupplyBean, dl: DataLoader):
    B.model.eval()

    meanloss = utils.AverageMeter()
    meanrmse = utils.AverageMeter()
    results = {"image": [], "loss": [], "rmse": []}
    with tqdm(total=len(dl.dataset), ncols=100, leave=False, file=sys.stdout, desc=f"Valid[{epoch:03d}]") as t:
        for f, x, offset, ratio, y in dl:
            x_ = x.cuda()
            y_ = y.cuda()
            p_ = B.model(x_)
            loss = B.criterion(p_, y_)
            rmse = B.criterion_rmse(p_, y_, ratio.cuda())

            meanloss.update(loss.item())
            meanrmse.update(rmse.item())
            results["image"].append(f)
            results["loss"].append(loss.item())
            results["rmse"].append(rmse.item())
            t.set_postfix_str(f"loss: {loss.item():.6f}, rmse: {rmse.item():.6f}", refresh=False)
            t.update(len(x))

    return meanloss(), meanrmse(), results


""" 
=================================================================================
Training with KFold
================================================================================= 
"""


kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
kflist = list(kf.split(ds_train_total))
for fold in FOLDS:
    train_idx, valid_idx = kflist[fold]
    ds_train = Subset(ds_train_total, train_idx)
    ds_valid = Subset(ds_train_total, valid_idx)
    dl_train = DataLoader(ds_train, **dl_kwargs, shuffle=True)
    dl_valid = DataLoader(ds_valid, **dl_kwargs, shuffle=False)
    B = SupplyBean()
    torch.cuda.empty_cache()

    for epoch in range(START_EPOCH, 100):
        tloss, trmse, tres = train_loop(B, dl_train)
        vloss, vrmse, vres = valid_loop(B, dl_valid)

        # Logging
        log.info(f"Epoch: {epoch:03d}, loss: {tloss:.6f};{vloss:.6f}, rmse {trmse:.6f};{vrmse:.6f}")
        B.scheduler.step(vloss)

        # Earlystop
        if vloss < best_loss:
            best_loss = vloss
            early_stop_cnt = 0

            torch.save(
                {
                    "model": B.model.state_dict(),
                    "optimizer": B.optimizer.state_dict(),
                    "epoch": epoch,
                },
                RESULT_DIR / f"ckpt-{UID}_{fold}.pth",
            )
        elif early_stop_cnt >= 10:
            log.info(f"Stop training at epoch {epoch} of fold {fold}.")
            break
        else:
            early_stop_cnt += 1
