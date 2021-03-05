"""
# 할 것

* 일단 돌려서 submission 만들어보기

* 데이터 augmentation 어떻게 하는지? flip 정도 가능할듯?
https://blog.paperspace.com/data-augmentation-for-bounding-boxes/
shift + rotation + scale 모두 어느정도 도움이 될 수 있겠다.
근데 scale은 이미지 사이즈를 안 바꾸는 방법으로 했으면 좋겠는데;;

* 이미지를 필요 영역만 잘라서 넣어줬을 때 어떻게 되는지?*

* RMSE metric 추가
* validation의 0번째 아이템으로 예제 이미지 추가
* 마지막 convtranspose weight를 처음에 설정하는게 좋은지 아닌지
* keypoint loss만으로 학습했을 때?
"""

import math
import os
import random
from collections import defaultdict
from datetime import datetime
from io import TextIOWrapper
from pathlib import Path
from typing import Callable, List, Sequence, Tuple
import shutil

import albumentations as A
import cv2
import imageio
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
from torchvision.models import mobilenet_v2
from torchvision.models.detection import KeypointRCNN, keypointrcnn_resnet50_fpn
from torchvision.ops import MultiScaleRoIAlign
from tqdm import tqdm

import utils

BASELINE = True
MODEL = "keypointrcnn_resnet50_fpn_finetune"
DATA_DIR = Path("data/ori")
FOLD = 1
NUM_FOLDS = 5
SAM = False
LR = 1e-4
BATCH_SIZE = 10
SEED = 20210304

START_EPOCH = 1
FINETUNE_EPOCH = 30
NUM_EPOCHS = 200
if BASELINE:
    NUM_EPOCHS = 10
    FINETUNE_EPOCH = 5

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
shutil.copy("main-baseline.py", EXPATH / "main-baseline.py")
print(EXNAME)
utils.seed_everything(SEED)


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
        torch.cuda.empty_cache()

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

            if self.epoch == FINETUNE_EPOCH:
                for p in self.model.parameters():
                    p.requires_grad = True

    def train_loop(self):
        self.model.train()

        self.tloss = {
            "total_loss": utils.AverageMeter(),
            "loss_classifier": utils.AverageMeter(),
            "loss_box_reg": utils.AverageMeter(),
            "loss_keypoint": utils.AverageMeter(),
            "loss_objectness": utils.AverageMeter(),
            "loss_rpn_box_reg": utils.AverageMeter(),
        }
        with tqdm(total=len(self.dl_train.dataset), ncols=100, leave=False, desc=f"{self.sepoch} train") as t:
            for files, xs, ys in self.dl_train:
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

                self.tloss["total_loss"].update(loss.item())
                self.tloss["loss_classifier"].update(losses["loss_classifier"])
                self.tloss["loss_box_reg"].update(losses["loss_box_reg"])
                self.tloss["loss_keypoint"].update(losses["loss_keypoint"])
                self.tloss["loss_objectness"].update(losses["loss_objectness"])
                self.tloss["loss_rpn_box_reg"].update(losses["loss_rpn_box_reg"])
                t.set_postfix_str(f"loss: {loss.item():.6f} keypoint: {losses['loss_keypoint']:.6f}", refresh=False)
                t.update(len(xs))

        self.tloss["total_loss"] = self.tloss["total_loss"]()
        self.tloss["loss_classifier"] = self.tloss["loss_classifier"]()
        self.tloss["loss_box_reg"] = self.tloss["loss_box_reg"]()
        self.tloss["loss_keypoint"] = self.tloss["loss_keypoint"]()
        self.tloss["loss_objectness"] = self.tloss["loss_objectness"]()
        self.tloss["loss_rpn_box_reg"] = self.tloss["loss_rpn_box_reg"]()

    @torch.no_grad()
    def valid_loop(self):
        self.model.train()  # loss 구하기 위함...

        self.vloss = {
            "total_loss": utils.AverageMeter(),
            "loss_classifier": utils.AverageMeter(),
            "loss_box_reg": utils.AverageMeter(),
            "loss_keypoint": utils.AverageMeter(),
            "loss_objectness": utils.AverageMeter(),
            "loss_rpn_box_reg": utils.AverageMeter(),
        }
        with tqdm(total=len(self.dl_valid.dataset), ncols=100, leave=False, desc=f"{self.sepoch} valid") as t:
            for files, xs, ys in self.dl_valid:
                xs_ = [x.cuda() for x in xs]
                ys_ = [{k: v.cuda() for k, v in y.items()} for y in ys]
                losses = self.model(xs_, ys_)
                loss = sum(loss for loss in losses.values())

                self.vloss["total_loss"].update(loss.item())
                self.vloss["loss_classifier"].update(losses["loss_classifier"])
                self.vloss["loss_box_reg"].update(losses["loss_box_reg"])
                self.vloss["loss_keypoint"].update(losses["loss_keypoint"])
                self.vloss["loss_objectness"].update(losses["loss_objectness"])
                self.vloss["loss_rpn_box_reg"].update(losses["loss_rpn_box_reg"])
                t.set_postfix_str(f"loss: {loss.item():.6f} keypoint: {losses['loss_keypoint']:.6f}", refresh=False)
                t.update(len(xs))

        self.vloss["total_loss"] = self.vloss["total_loss"]()
        self.vloss["loss_classifier"] = self.vloss["loss_classifier"]()
        self.vloss["loss_box_reg"] = self.vloss["loss_box_reg"]()
        self.vloss["loss_keypoint"] = self.vloss["loss_keypoint"]()
        self.vloss["loss_objectness"] = self.vloss["loss_objectness"]()
        self.vloss["loss_rpn_box_reg"] = self.vloss["loss_rpn_box_reg"]()

    @torch.no_grad()
    def callback(self):
        # TODO RMSE metric
        fepoch = self.fold * 1000 + self.epoch
        now = datetime.now()
        msg = (
            f"[{now.month:02d}:{now.day:02d}-{now.hour:02d}:{now.minute:02d} {self.sepoch}:{self.fold}] "
            f"loss [{self.tloss['total_loss']:.6f}:{self.vloss['total_loss']:.6f}] "
            f"classifier [{self.tloss['loss_classifier']:.6f}:{self.vloss['loss_classifier']:.6f}] "
            f"box_reg [{self.tloss['loss_box_reg']:.6f}:{self.vloss['loss_box_reg']:.6f}] "
            f"keypoint [{self.tloss['loss_keypoint']:.6f}:{self.vloss['loss_keypoint']:.6f}] "
            f"objectness [{self.tloss['loss_objectness']:.6f}:{self.vloss['loss_objectness']:.6f}] "
            f"rpn_box_reg [{self.tloss['loss_rpn_box_reg']:.6f}:{self.vloss['loss_rpn_box_reg']:.6f}] "
        )
        print(msg)
        self.logger.write(msg + "\r\n")
        self.logger.flush()

        # Tensorboard
        loss_scalars = {"t": self.tloss["total_loss"], "v": self.vloss["total_loss"]}
        loss_classifier = {"t": self.tloss["loss_classifier"], "v": self.vloss["loss_classifier"]}
        loss_box_reg = {"t": self.tloss["loss_box_reg"], "v": self.vloss["loss_box_reg"]}
        loss_keypoint = {"t": self.tloss["loss_keypoint"], "v": self.vloss["loss_keypoint"]}
        loss_objectness = {"t": self.tloss["loss_objectness"], "v": self.vloss["loss_objectness"]}
        loss_rpn_box_reg = {"t": self.tloss["loss_rpn_box_reg"], "v": self.vloss["loss_rpn_box_reg"]}
        self.writer.add_scalars(self.exname + "/loss", loss_scalars, fepoch)
        self.writer.add_scalars(self.exname + "/loss_classifier", loss_classifier, fepoch)
        self.writer.add_scalars(self.exname + "/loss_box_reg", loss_box_reg, fepoch)
        self.writer.add_scalars(self.exname + "/loss_keypoint", loss_keypoint, fepoch)
        self.writer.add_scalars(self.exname + "/loss_objectness", loss_objectness, fepoch)
        self.writer.add_scalars(self.exname + "/loss_rpn_box_reg", loss_rpn_box_reg, fepoch)

        # LR scheduler
        self.scheduler.step(self.vloss["total_loss"])

        if self.best_loss - self.vloss["total_loss"] > 1e-8:
            self.best_loss = self.vloss["total_loss"]
            self.earlystop_cnt = 0

            # Save Checkpoint
            state_dict = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": self.epoch,
                "SAM": SAM,
            }
            torch.save(state_dict, self.expath / f"best-ckpt-{self.fold}.pth")
        else:
            self.earlystop_cnt += 1

    @torch.no_grad()
    def submission(self, dl, submission_df: pd.DataFrame):
        torch.cuda.empty_cache()

        # load best checkpoint
        ckpt_path = EXPATH / f"best-ckpt-{self.fold}.pth"
        print("Load best checkpoint", ckpt_path)
        self.model.load_state_dict(torch.load(ckpt_path)["model"])
        self.model.eval()

        # reverse_transform = A.Compose([]) # TODO resize가 있으면 쓰는 것도 좋을듯

        output_files = []
        output_keypoints = []
        with tqdm(total=len(dl.dataset), ncols=100, leave=False, desc=f"Submission:{self.fold}") as t:
            for files, xs in dl:
                xs_ = [x.cuda() for x in xs]
                results_ = self.model(xs_)
                for file, result_ in zip(files, results_):
                    keypoints_ = result_["keypoints"][0, :, :2]
                    # 이미지를 왼쪽 400, 위쪽 100만큼 잘라냈으므로 보상해줌
                    keypoints_[:, 0] += 400
                    keypoints_[:, 1] += 100

                    output_files.append(file.name)
                    output_keypoints.append(keypoints_.cpu().flatten().numpy())
        output_files = np.array(output_files)
        output_keypoints = np.stack(output_keypoints)
        output = np.concatenate([output_files, output_keypoints], axis=1)
        output = pd.DataFrame(output, columns=submission_df.columns)
        output.to_csv(self.expath / f"submission-{self.fold}.csv", index=False)


class KeypointDataset(Dataset):
    def __init__(self, image_files: os.PathLike, df: os.PathLike = None, transforms: Sequence[Callable] = None):
        self.image_files = image_files
        self.df = df
        self.transforms = transforms

    def __len__(self) -> int:
        return self.df.shape[0]

    def __getitem__(self, index: int):
        image_file = self.image_files[index]
        image = imageio.imread(image_file)

        if self.df is not None:
            labels = np.array([1])
            keypoints = self.df[index, 1:].reshape(-1, 2).astype(np.int64)
            x1, y1 = min(keypoints[:, 0]), min(keypoints[:, 1])
            x2, y2 = max(keypoints[:, 0]), max(keypoints[:, 1])
            boxes = np.array([[x1, y1, x2, y2]], dtype=np.int64)

            targets = {
                "image": image,
                "bboxes": boxes,
                "labels": labels,
                "keypoints": keypoints,
            }
            targets = self.transforms(**targets)

            image = targets["image"]
            # image = image / 255.0 # 255으로 나눠주면 안됨. ToTensorV2에서 255로 나눠주나봄?

            targets = {
                "labels": torch.as_tensor(targets["labels"], dtype=torch.int64),
                "boxes": torch.as_tensor(targets["bboxes"], dtype=torch.float32),
                "keypoints": torch.as_tensor(
                    np.concatenate([targets["keypoints"], np.ones((24, 1))], axis=1)[np.newaxis],
                    dtype=torch.float32,
                ),
            }

            return image_file, image, targets
        else:
            targets = self.transforms(image=image)
            image = targets["image"]

            return image_file, image


def collate_fn(batch: torch.Tensor) -> Tuple:
    return tuple(zip(*batch))


def load_dataset(fold):
    transform_train = A.Compose(
        [
            # Crop으로 이미지 크기는 1100, 900이 되는 것
            A.Crop(400, 100, 1500, 1000),
            # TODO 이미 1100x900 이므로 resize는 하지 않는게 좋을지 확인해봐야함
            # MMdetection 등에서는 800, 1333을 쓰니까, 그 사이즈가 더 유리할 수도 있다고는 생각된다.
            # A.Resize(800, 1333),
            A.HorizontalFlip(),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        keypoint_params=A.KeypointParams(format="xy"),
    )
    transform_valid = A.Compose(
        [
            # Crop으로 이미지 크기는 1100, 900이 되는 것
            A.Crop(400, 100, 1500, 1000),
            # TODO 이미 1100x900 이므로 resize는 하지 않는게 좋을지 확인해봐야함
            # MMdetection 등에서는 800, 1333을 쓰니까, 그 사이즈가 더 유리할 수도 있다고는 생각된다.
            # A.Resize(800, 1333),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        keypoint_params=A.KeypointParams(format="xy"),
    )
    transform_test = A.Compose(
        [
            A.Crop(400, 100, 1500, 1000),
            # A.Resize(800, 1333),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
    )

    train_files = np.array(sorted(list((DATA_DIR / "train_imgs").glob("*.jpg"))))
    test_files = np.array(sorted(list((DATA_DIR / "test_imgs").glob("*.jpg"))))
    df = pd.read_csv(DATA_DIR / "train_df.csv").to_numpy()

    ds_test = KeypointDataset(test_files, transforms=transform_test)
    dl_kwargs = dict(batch_size=BATCH_SIZE, num_workers=4, collate_fn=collate_fn)
    dl_test = DataLoader(ds_test, **dl_kwargs, shuffle=False)

    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=1351235)
    tidx, vidx = list(kf.split(train_files))[fold - 1]

    # BASELINE 모드일 때는 5%의 데이터만 써서 학습
    if BASELINE:
        np.random.seed(SEED)
        tidx = np.random.choice(tidx, 168)
        # validation은 줄이지 않음 --> 성능 비교 확실하게 하기 위해
        # vidx = np.random.choice(vidx, 42)

    tds = KeypointDataset(train_files[tidx], df[tidx], transforms=transform_train)
    vds = KeypointDataset(train_files[vidx], df[vidx], transforms=transform_valid)
    tdl = DataLoader(tds, **dl_kwargs, shuffle=True)
    vdl = DataLoader(vds, **dl_kwargs, shuffle=False)
    return tdl, vdl


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
    elif MODEL == "keypointrcnn_resnet50_fpn_finetune":
        model = keypointrcnn_resnet50_fpn(pretrained=True, progress=False)
        for p in model.parameters():
            p.requires_grad = False

        m = nn.ConvTranspose2d(512, 24, 4, 2, 1)
        with torch.no_grad():
            m.weight[:, :17] = model.roi_heads.keypoint_predictor.kps_score_lowres.weight
            m.bias[:17] = model.roi_heads.keypoint_predictor.kps_score_lowres.bias
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
