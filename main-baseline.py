"""
# 할 것

* 일단 돌려서 submission 만들어보기
* 데이터 augmentation 어떻게 하는지? flip 정도 가능할듯?
https://blog.paperspace.com/data-augmentation-for-bounding-boxes/
rotation은 데이터를 보면 필요 없어보이기도 하는데, 일반화 성능이 증가할지도 모르므로?
* 이미지를 필요 영역만 잘라서 넣어줬을 때 어떻게 되는지?
* RMSE metric 추가
* validation의 0번째 아이템으로 예제 이미지 추가
* transform 돌렸을 때 어떻게 변화되는가 그려보면서 확인
"""

import math
import os
import random
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
from torch.utils.data.dataset import Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import mobilenet_v2
from torchvision.models.detection import KeypointRCNN, keypointrcnn_resnet50_fpn
from torchvision.ops import MultiScaleRoIAlign
from tqdm import tqdm

import utils

BASELINE = True
MODEL = "keypointrcnn_resnet50_fpn_finetune_step1"
DATA_DIR = Path("data/ori")
FOLD = 1
START_EPOCH = 1
NUM_EPOCHS = 200
N_TTA_TEST = 10
N_TTA_VALID = 1
SAM = False
LR = 1e-4

CKPT = None
LOG_DIR = Path("log" + ("/baseline" if BASELINE else ""))
RESULT_DIR = Path("results" + ("/baseline" if BASELINE else ""))
COMMENTS = [
    MODEL,
    "SAM" if SAM else None,
    f"LR{LR}",
    f"fold{FOLD}",
    "baseline" if BASELINE else None,
]
EXPATH, EXNAME = utils.generate_experiment_directory(RESULT_DIR, COMMENTS)
print(EXNAME)
utils.seed_everything(20210304)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: nn.Module,
        writer: SummaryWriter,
        logger: TextIOWrapper,
        fold: int,
        expath: Path,
        exname: str,
    ):
        self.model = model
        self.optimizer = optimizer
        self.writer = writer
        self.logger = logger
        self.fold = fold
        self.expath = expath
        self.exname = exname

    def fit(self, dl_train, dl_valid, num_epochs, start_epoch=1):
        self.earlystop_cnt = 0
        self.best_loss = math.inf
        self.dl_train = dl_train
        self.dl_valid = dl_valid

        # scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            factor=0.5,
            patience=3,
            verbose=True,
            threshold=1e-8,
            cooldown=0,
        )

        for self.epoch in range(start_epoch, num_epochs + 1):
            self.sepoch = f"[{self.epoch:03d}/{num_epochs}]"
            self.train_loop()
            self.valid_loop()
            self.callback()

            if self.earlystop_cnt > 10:
                print("[Early Stop]", self.exname)
                break

    def train_loop(self):
        self.model.train()

        self.tloss = utils.AverageMeter()
        with tqdm(total=len(self.dl_train.dataset), ncols=100, leave=False, desc=f"{self.sepoch} train") as t:
            for xs, ys in self.dl_train:
                xs_ = [x.cuda() for x in xs]
                ys_ = [{k: v.cuda() for k, v in y.items()} for y in ys]
                losses = self.model(xs_, ys_)
                loss = sum(loss for loss in losses.values())

                self.optimizer.zero_grad()
                loss.backward()
                if SAM:
                    self.optimizer.first_step(zero_grad=True)
                    sum(self.model(xs_, ys_)).backward()
                    self.optimizer.second_step(zero_grad=False)
                else:
                    self.optimizer.step()

                self.tloss.update(loss.item())
                t.set_postfix_str(f"loss: {loss.item():.6f}", refresh=False)
                t.update(len(xs))

        self.tloss = self.tloss()

    @torch.no_grad()
    def valid_loop(self):
        self.model.eval()

        self.vloss = utils.AverageMeter()
        with tqdm(total=len(self.dl_valid.dataset), ncols=100, leave=False, desc=f"{self.sepoch} valid") as t:
            for xs, ys in self.dl_valid:
                xs_ = [x.cuda() for x in xs]
                ys_ = [{k: v.cuda() for k, v in y.items()} for y in ys]
                losses = self.model(xs_, ys_)
                loss = sum(loss for loss in losses.values())

                self.vloss.update(loss.item())
                t.set_postfix_str(f"loss: {loss.item():.6f}")
                t.update(len(xs))

        self.vloss = self.vloss()

    @torch.no_grad()
    def callback(self):
        # TODO RMSE metric
        fepoch = self.fold * 1000 + self.epoch
        now = datetime.now()
        msg = (
            f"[{now.month:02d}:{now.day:02d}-{now.hour:02d}:{now.minute:02d} {self.sepoch}:{self.fold}]"
            f"loss [{self.tloss:.6f}:{self.vloss:.6f}]"
        )
        print(msg)
        self.logger.write(msg + "\r\n")
        self.logger.flush()

        # Tensorboard
        loss_scalars = {"t": self.tloss, "v": self.vloss}
        self.writer.add_scalars(self.exname + "/loss", loss_scalars, fepoch)

        # LR scheduler
        self.scheduler.step(self.vloss)

        if self.best_loss - self.vloss > 1e-8:
            self.best_loss = self.vloss
            self.earlystop_cnt = 0

            # Save Checkpoint
            state_dict = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": self.epoch,
                "SAM": SAM,
            }
            torch.save(state_dict, self.expath / f"best-ckpt-{self.fold}.pth")

            # Save probabilities
            np.savez_compressed(
                EXPATH / f"prob-{self.fold}.npz",
                tps=self.tps,
                tys=self.tys,
                vps=self.vps_ori,
                vys=self.vys,
            )
        else:
            self.earlystop_cnt += 1

    @torch.no_grad()
    def submission(self, dl):
        # load best checkpoint
        ckpt_path = EXPATH / f"best-ckpt-{self.fold}.pth"
        print("Load best checkpoint", ckpt_path)
        self.model.load_state_dict(torch.load(ckpt_path)["model"])

        # TODO

        pass


class KeypointDataset(Dataset):
    def __init__(
        self,
        image_dir: os.PathLike,
        label_path: os.PathLike,
        transforms: Sequence[Callable] = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.df = pd.read_csv(label_path).to_numpy()
        self.transforms = transforms

    def __len__(self) -> int:
        return self.df.shape[0]

    def __getitem__(self, index: int):
        image_id = self.df[index, 0]
        labels = np.array([1])
        # int64가 아니면 안되는건가? 소숫점이 손실될텐데?
        keypoints = self.df[index, 1:].reshape(-1, 2).astype(np.int64)

        x1, y1 = min(keypoints[:, 0]), min(keypoints[:, 1])
        x2, y2 = max(keypoints[:, 0]), max(keypoints[:, 1])
        boxes = np.array([[x1, y1, x2, y2]], dtype=np.int64)

        image = cv2.imread(str(self.image_dir / image_id), cv2.COLOR_BGR2RGB)

        targets = {
            "image": image,
            "bboxes": boxes,
            "labels": labels,
            "keypoints": keypoints,
        }

        if self.transforms is not None:
            targets = self.transforms(**targets)

        image = targets["image"]
        image = image / 255.0

        targets = {
            "labels": torch.as_tensor(targets["labels"], dtype=torch.int64),
            "boxes": torch.as_tensor(targets["bboxes"], dtype=torch.float32),
            "keypoints": torch.as_tensor(
                np.concatenate([targets["keypoints"], np.ones((24, 1))], axis=1)[np.newaxis],
                dtype=torch.float32,
            ),
        }

        return image, targets


def collate_fn(batch: torch.Tensor) -> Tuple:
    return tuple(zip(*batch))


def load_dataset(fold):
    transform = A.Compose(
        [
            A.Resize(800, 1333),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        keypoint_params=A.KeypointParams(format="xy"),
    )

    ds = KeypointDataset(DATA_DIR / "train_imgs", DATA_DIR / "train_df.csv", transform)
    kf = KFold(n_splits=5, shuffle=True, random_state=1351235)
    for i, (tidx, vidx) in enumerate(kf.split(ds), 1):
        if i == fold:
            tds, vds = Subset(ds, tidx), Subset(ds, vidx)
            tdl = DataLoader(tds, batch_size=24, shuffle=True, num_workers=8, pin_memory=True, collate_fn=collate_fn)
            vdl = DataLoader(vds, batch_size=24, shuffle=False, num_workers=8, pin_memory=True, collate_fn=collate_fn)
            return tdl, vdl

    raise NotImplementedError("out of folds")


def get_model() -> nn.Module:
    if MODEL == "우주대마왕":
        backbone = mobilenet_v2(pretrained=True).features
        backbone.out_channels = 1280
        roi_pooler = MultiScaleRoIAlign(featmap_names=["0"], output_size=7, sampling_ratio=2)

        keypoint_roi_pooler = MultiScaleRoIAlign(featmap_names=["0"], output_size=14, sampling_ratio=2)

        model = KeypointRCNN(
            backbone,
            num_classes=2,
            num_keypoints=24,
            box_roi_pool=roi_pooler,
            keypoint_roi_pool=keypoint_roi_pooler,
        )
    elif MODEL == "keypointrcnn_resnet50_fpn_finetune_step1":
        model = keypointrcnn_resnet50_fpn(pretrained=True, progress=False)
        for p in model.parameters():
            p.requires_grad = False

        m = nn.ConvTranspose2d(512, 24, 4, 2, 1)
        with torch.no_grad():
            m.weight[:, :17] = model.roi_heads.keypoint_predictor.kps_score_lowres.weight
            m.bias[:17] = model.roi_heads.keypoint_predictor.kps_score_lowres.bias
            # m.weight = m.weight.contiguous()
            # m.bias = m.bias.contiguous()
        model.roi_heads.keypoint_predictor.kps_score_lowres = m
    else:
        raise NotImplementedError()

    return model.cuda()


def main():
    model = get_model()

    if SAM:
        optimizer = utils.SAM(model.parameters(), optim.AdamW, lr=1e-4)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    start_epoch = 1
    if CKPT is not None and Path(CKPT).exists():
        ckpt = torch.load(CKPT)
        model.load_state_dict(ckpt["model"])
        if ckpt["SAM"] == SAM:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]

    writer = SummaryWriter(LOG_DIR)
    logger = open(EXPATH / f"log-{FOLD}.log", "w")
    tdl, vdl = load_dataset(FOLD)
    trainer = Trainer(model, optimizer, writer, logger, FOLD, EXPATH, EXNAME)
    trainer.fit(tdl, vdl, NUM_EPOCHS, start_epoch)
    logger.close()


if __name__ == "__main__":
    main()
