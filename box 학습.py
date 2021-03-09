#!/usr/bin/env python
# coding: utf-8

# HRNet이 384x288이지만, 지금 영상이 고화질이다보니 대부분 사람 크기가 그 두배 좀 안됨.
# 그러므로 사람을 그 두배인 768x576크기로 만들자.
# 하지만 가급적이면 비율이 맞도록

# ---
# 
# ## 라이브러리 로딩

# In[1]:


#get_ipython().run_line_magic('load_ext', 'lab_black')


# In[27]:


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
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import KFold
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from tqdm import tqdm

import utils
import networks


# ---
# 
# ## 하이퍼 파라미터

# In[3]:


RESULT_DIR = Path("results/box학습")


# In[4]:


LR = 1e-4  # transfer learning이니깐 좀 작게 주는게 좋을 것 같아서 1e-4
BATCH_SIZE = 10
START_EPOCH = 1


# In[5]:


n = datetime.now()
UID = f"{n.year:04d}{n.month:02d}{n.day:02d}-{n.hour:02d}{n.minute:02d}{n.second:02d}"
SEED = 20210309


# In[6]:


utils.seed_everything(SEED, deterministic=False)

RESULT_DIR.mkdir(parents=True, exist_ok=True)
log = utils.CustomLogger(RESULT_DIR / f"log_{UID}.log", "w")
log.info("학습 시작")


# ---
# 
# ## 데이터 로딩

# 시간이 많지 않으니 box는 CrossValidation하지 않고, 대신 fold만 10개로 나눠줌

# In[7]:


train_imgs_ori = np.array(sorted(list(Path("data/ori/train_imgs/").glob("*.jpg"))))
test_imgs = np.array(sorted(list(Path("data/ori/test_imgs/").glob("*.jpg"))))
train_df = pd.read_csv("data/ori/train_df.csv")


# In[8]:


kf = KFold(n_splits=10, shuffle=True, random_state=SEED)


# In[9]:


train_idx, valid_idx = next(kf.split(train_imgs_ori).__iter__())


# In[10]:


train_imgs = train_imgs_ori[train_idx]
valid_imgs = train_imgs_ori[valid_idx]


# In[11]:


dfn = train_df.to_numpy()
train_keypoints = dfn[train_idx, 1:].reshape(len(train_idx), -1, 2)
valid_keypoints = dfn[valid_idx, 1:].reshape(len(valid_idx), -1, 2)


# In[12]:


log.info(f"train: {train_keypoints.shape}, valid: {valid_keypoints.shape}, test: {len(test_imgs)}")


# train 3397, valid 378개, test 1600개

# In[13]:


class ImageDataset(Dataset):
    def __init__(self, files, keypoints=None, padding=40):
        super().__init__()
        self.files = files
        self.keypoints = keypoints
        self.padding = padding

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        img = imageio.imread(f)
        x = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

        if self.keypoints is not None:
            keypoints = self.keypoints[idx]
            xmin = keypoints[:, 0].min() - self.padding
            xmax = keypoints[:, 0].max() + self.padding
            ymin = keypoints[:, 1].min() - self.padding
            ymax = keypoints[:, 1].max() + self.padding
            target = {
                "labels": torch.tensor([1], dtype=torch.int64),
                "boxes": torch.tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32),
            }
            return x, target
        return x


# In[14]:


ds_train = ImageDataset(train_imgs, train_keypoints)
ds_valid = ImageDataset(valid_imgs, valid_keypoints)
ds_test = ImageDataset(test_imgs)


# In[15]:


print("data example:\r\n", str(ds_train[0][0]), "\r\n", str(ds_train[0][1]))


# In[16]:


collate_fn = lambda x: tuple(zip(*x))
dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, num_workers=4, collate_fn=collate_fn, shuffle=True)
dl_valid = DataLoader(ds_valid, batch_size=BATCH_SIZE, num_workers=4, collate_fn=collate_fn, shuffle=False)
dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)


# ---
# 
# ## 학습

# In[17]:


box_model = fasterrcnn_resnet50_fpn(pretrained=True, progress=False).cuda()


# In[18]:


optimizer = optim.AdamW(box_model.parameters(), lr=LR)


# In[19]:


scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)


# In[20]:


def train_loop(dl: DataLoader):
    torch.cuda.empty_cache()
    box_model.train()

    meanloss = utils.AverageMeter()
    with tqdm(total=len(dl.dataset), ncols=100, leave=False, file=sys.stdout) as t:
        for xs, ys in dl:
            xs_ = [x.cuda() for x in xs]
            ys_ = [{k: v.cuda() for k, v in y.items()} for y in ys]
            losses = box_model(xs_, ys_)
            loss = sum(loss for loss in losses.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            meanloss.update(loss.item())
            t.set_postfix_str(f"loss: {loss.item():.6f}", refresh=False)
            t.update(len(xs))

    return meanloss()


# In[21]:


@torch.no_grad()
def valid_loop(dl: DataLoader):
    torch.cuda.empty_cache()
    box_model.train()

    meanloss = utils.AverageMeter()
    with tqdm(total=len(dl.dataset), ncols=100, leave=False, file=sys.stdout) as t:
        for xs, ys in dl:
            xs_ = [x.cuda() for x in xs]
            ys_ = [{k: v.cuda() for k, v in y.items()} for y in ys]
            losses = box_model(xs_, ys_)
            loss = sum(loss for loss in losses.values())

            meanloss.update(loss.item())
            t.set_postfix_str(f"val_loss: {loss.item():.6f}", refresh=False)
            t.update(len(xs))

    return meanloss()


# In[22]:


best_loss = math.inf
early_stop_cnt = 0

for epoch in range(START_EPOCH, 999):
    tloss = train_loop(dl_train)
    vloss = valid_loop(dl_valid)
    
    # Logging
    log.info(f'Epoch: {epoch:03d}, loss: {tloss:.6f} ; {vloss:.6f}')
    scheduler.step(vloss)
    
    # Earlystop
    if vloss < best_loss:
        best_loss = vloss
        early_stop_cnt = 0
        
        torch.save({
            'model': box_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }, RESULT_DIR/f'ckpt-{UID}.pth')
    elif early_stop_cnt >= 20:
        log.info(f'Stop training at epoch {epoch}.')
        break
    else:
        early_stop_cnt +=1


# best ckpt 불러와서 ds_train, ds_valid 합쳐서 2epoch정도 더 학습하기

# In[24]:


ckpt = torch.load(RESULT_DIR / f"ckpt-{UID}.pth")
box_model.load_state_dict(ckpt["model"])


# Pytorch의 ChainDataset이 shuffle이 안되서 그냥 직접 만들음

# In[28]:


class ChainDataset(Dataset):
    def __init__(self, *ds_list: Dataset):
        """
        Combine multiple dataset into one.
        Parameters
        ----------
        ds_list: list of datasets
        """
        self.ds_list = ds_list
        self.len_list = [len(ds) for ds in self.ds_list]
        self.total_len = sum(self.len_list)

        self.idx_list = []
        for i, l in enumerate(self.len_list):
            self.idx_list.extend([(i, j) for j in range(l)])

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        didx, sidx = self.idx_list[idx]
        return self.ds_list[didx][sidx]


# In[30]:


ds_total = ChainDataset(ds_train, ds_valid)
dl_total = DataLoader(ds_total, batch_size=BATCH_SIZE, num_workers=4, collate_fn=collate_fn, shuffle=True)


# In[31]:


train_loop(dl_total)


# HRNet이 고정된 이미지 사이즈를 받는지를 우선 확인해보는게 먼저일듯?
