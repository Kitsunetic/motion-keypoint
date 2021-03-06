"""
# 이미 한 것

- 일단 돌려서 submission 만들어보기
- 이미지 resize한게 더 좋음
- Crop을 안함
- finetune 말고 처음부터


====================================================================
# 결과

* 이미지 resize한게 더 좋음
    - non baseline: 1.807918:2.240550 --> 1.552490:2.205956 (41.7725228481)
    - _on bsaeline: 2.844641:3.861035 --> 2.894984:3.741140

* Crop을 안함
    - 2.894984:3.741140 --> 2.625810:4.103032

* finetune 말고 처음부터
    2.894984:3.741140 --> 2.924908:3.780813
    금방 학습이 되기는 하나, 쉽게 과적합하는듯?
    최종: 1.877865:5.026970

* Rotation. limit=10
    처음에는 학습이 조금 느려지는거 같기도?
    2.894984:3.741140 --> 2.856512:3.611753
    0.13만큼 좋아짐

* Rotation. limit=20
    2.894984:3.741140 --> 2.966834:3.590582
    0.15만큼 좋아짐

* 1300x1030
    padding이 작아서 rotation할 때 keypoint가 삭제되는 문제 때문에 crop 영역을 좀 더 넓게 잡아줌
    2.894984:3.741140 --> 3.166891:3.731480
    
* Rotation. limit=10
    3.166891:3.731480 --> 2.662445:3.811243
    
* Rotation. limit=20
    3.166891:3.731480 --> 3.185518:3.806415
    
* Rotation. limit=30
    3.166891:3.731480 --> 2.933816:3.737933

* Crop 1100x1030
    3.166891:3.731480 --> 3.058709:3.648326
    crop 중에선 가장 좋았음

* Crop 1100x1030 + rotate10
    3.058709:3.648326 --> 3.233407:3.688290
    약간 나빠지네?

* Crop 1100x1030 + rotate20
    3.058709:3.648326 --> 2.808439:3.732426

* Crop 1100x1030 + rotate30 (오류)
* Crop 1100x1030 + rotate25
    3.058709:3.648326 --> 3.172204:3.614873
* Crop 1100x1030 + rotate25
    3.058709:3.648326 --> 3.170852:3.649939

* Crop 1100x1030 + rotate25 + shift0.01
    3.170852:3.649939 --> 2.988702:3.645297
* Crop 1100x1030 + rotate25 + shift0.02
    3.170852:3.649939 --> 2.902854:3.772051
    shift는 오히려 나빠지기만 하는듯? rotate도 그닥 큰 효과는 없는거같고

* Crop 1100x900 + rotate5 + shift0.02
    2.902854:3.772051 --> 3.405847:3.727300 
* Crop 1100x900 + rotate5 + shift0.03
    2.902854:3.772051 --> 3.038964:3.756478 

* Crop 1100x900 + rotate15 + shift0.02
    2.902854:3.772051 --> 3.070648:3.690815 (*)
* Crop 1100x900 + rotate25 + shift0.02
    (오류)

* Crop 1100x900 + rotate15 + shift0.02
    3.070648:3.690815 == 3.087706:3.716364
* Crop 1100x900 + rotate15 + shift0.02 - hoizontal_flip
    3.070648:3.690815 --> 2.560510:3.427309 (horizontal flip은 빼는게 좋다?)

* Crop 1100x900 + rotate15 + shift0.02 + blur7
    3.145380:3.629517 blur가 도움이 되었다?
* Crop 1100x900 + rotate15 + shift0.02 + gamma80_120
* Crop 1100x900 + rotate15 + shift0.02 + brightness0.2

// 데이터를 2배로 늘리고

* Crop 1100x900 + rotate15 + shift0.02
    2.639951:3.297648
* Crop 1100x900 + rotate15 + shift0.02 - hoizontal_flip
    2.701766:3.019309
* Crop 1100x900 + rotate15 + shift0.02 + blur7
    2.506098:3.252584
* Crop 1100x900 + rotate15 + shift0.02 + gamma80_120
    2.359201:3.329015
* Crop 1100x900 + rotate15 + shift0.02 + brightness0.2
    2.590074:3.251132

* Crop 1100x900 + rotate15 + shift0.02 + contrast0.2
    2.468286:3.318427
* Crop 1100x900+shift0.02-hoizontal_flip+blur7+gamma80_120+brightness0.2 baseline
    2.421699:3.144095
* Crop 1100x900+shift0.02-hoizontal_flip+blur7+gamma80_120+brightness0.2 common (s3)
* Crop 1100x900+shift0.02-hoizontal_flip+blur7+gamma80_120+brightness0.2+contrast0.2 common (s5)





* efficientdet 모델 구조 / 입출력 확인

* RMSE metric 추가
* validation의 0번째 아이템으로 예제 이미지 추가
* 마지막 convtranspose weight를 처음에 설정하는게 좋은지 아닌지
* keypoint loss만으로 학습했을 때?

* GeneralRCNN? 에서 loss 구하는 코드 구해서 따로 로스 구하는 식 만들어주기
값이랑 loss를 둘 다 구할 수 있으면 좋겠다.

* augmentation들
    * shift
    * scale
    * contrast

* efficientdet
* scaled-yolo v4, v5
* detectors

* group normalization


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
    NUM_EPOCHS = 40
    FINETUNE_EPOCH = 15

CKPT = None
LOG_DIR = Path("log" + ("/baseline2" if BASELINE else "/_"))
RESULT_DIR = Path("results" + ("/baseline2" if BASELINE else "/_"))
COMMENTS = [
    MODEL,
    "SAM" if SAM else None,
    f"LR{LR}",
    f"fold{FOLD}",
    "baseline" if BASELINE else None,
    "1100x900+rotate15+shift0.02-hoizontal_flip+blur7+gamma80_120+brightness0.2",
]
EXPATH, EXNAME = utils.generate_experiment_directory(RESULT_DIR, COMMENTS)
shutil.copy("main-baseline.py", EXPATH / "main-baseline.py")
shutil.copy("utils.py", EXPATH / "utils.py")
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
        self.finetune_step = 1

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
            self.sepoch = f"[{self.epoch:03d}/{num_epochs:03d}]"
            self.train_loop()
            self.valid_loop()
            self.callback()

            if self.earlystop_cnt > 10:
                print("[Early Stop]", self.exname)
                break

            if self.finetune_step == 1 and self.epoch >= FINETUNE_EPOCH:
                self.finetune_step = 2
                self.earlystop_cnt = 0
                torch.cuda.empty_cache()
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
        # TODO loss가 아니라 RMSE를 구하도록 하는건?
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

        output_files = []
        output_keypoints = []
        with tqdm(total=len(dl.dataset), ncols=100, leave=False, desc=f"Submission:{self.fold}") as t:
            for files, xs in dl:
                xs_ = [x.cuda() for x in xs]
                results_ = self.model(xs_)
                for file, result_ in zip(files, results_):
                    keypoints = result_["keypoints"][0, :, :2].cpu().numpy()
                    # 이미지를 왼쪽 400, 위쪽 100만큼 잘라냈으므로 보상해줌
                    keypoints[:, 0] = keypoints[:, 0] * (1100 / xs[0].size(2)) + 400
                    keypoints[:, 1] = keypoints[:, 1] * (900 / xs[0].size(1)) + 100
                    # keypoints[:, 0] = keypoints[:, 0] * (1300 / xs[0].size(2)) + 300
                    # keypoints[:, 1] = keypoints[:, 1] * (1080 / xs[0].size(1)) + 50

                    output_files.append(file.name)
                    output_keypoints.append(keypoints.reshape(-1))

                    t.set_postfix_str(file.name, refresh=False)
                    t.update()
        output_files = np.array(output_files).reshape(-1, 1)
        output_keypoints = np.stack(output_keypoints)
        output = np.concatenate([output_files, output_keypoints], axis=1)
        output = pd.DataFrame(output, columns=submission_df.columns)
        output.to_csv(self.expath / f"submission-{self.exname}-{self.fold}.csv", index=False)


class KeypointDataset(Dataset):
    def __init__(self, image_files: os.PathLike, df: os.PathLike = None, transforms: Sequence[Callable] = None):
        self.image_files = image_files
        self.df = df
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.image_files)

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
            A.Crop(400, 100, 1500, 1000),  # 1100x900
            # A.Crop(300, 50, 1600, 1080),  # 1300x1030
            # A.Crop(400, 50, 1500, 1080),  # 1100x1030
            # MMDetection 등에서는 800x1333을 쓰니깐..?
            A.Resize(800, 1333),
            # A.HorizontalFlip(),
            A.Blur(blur_limit=7),
            A.RandomGamma(gamma_limit=(80, 120)),
            A.RandomBrightness(limit=0.2),
            # TODO contrast
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.0, rotate_limit=0),  # TODO no rotate
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        keypoint_params=A.KeypointParams(format="xy"),
    )
    transform_valid = A.Compose(
        [
            A.Crop(400, 100, 1500, 1000),  # 1100x900
            # A.Crop(300, 50, 1600, 1080),  # 1300x1030
            # A.Crop(400, 50, 1500, 1080),  # 1100x1030
            A.Resize(800, 1333),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        keypoint_params=A.KeypointParams(format="xy"),
    )
    transform_test = A.Compose(
        [
            A.Crop(400, 100, 1500, 1000),  # 1100x900
            # A.Crop(300, 50, 1600, 1080),  # 1300x1030
            # A.Crop(400, 50, 1500, 1080),  # 1100x1030
            A.Resize(800, 1333),
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
        tidx = np.random.choice(tidx, 336)
        # validation은 줄이지 않음 --> validation도 줄인다. 너무 오래걸려...
        vidx = np.random.choice(vidx, 84)

    tds = KeypointDataset(train_files[tidx], df[tidx], transforms=transform_train)
    vds = KeypointDataset(train_files[vidx], df[vidx], transforms=transform_valid)
    tdl = DataLoader(tds, **dl_kwargs, shuffle=True)
    vdl = DataLoader(vds, **dl_kwargs, shuffle=False)
    return tdl, vdl, dl_test


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
    submission_df = pd.read_csv(DATA_DIR / "train_df.csv")
    model = get_model()

    if SAM:
        optimizer = utils.SAM(model.parameters(), optim.AdamW, lr=LR)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=LR)

    start_epoch = 1
    if CKPT is not None and Path(CKPT).exists():
        ckpt = torch.load(CKPT)
        model.load_state_dict(ckpt["model"])
        if ckpt["SAM"] == SAM:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]

    writer = SummaryWriter(LOG_DIR)
    logger = open(EXPATH / f"log-{FOLD}.log", "w")
    logger.write(EXNAME + "\r\n")
    logger.flush()

    tdl, vdl, dl_test = load_dataset(FOLD)
    trainer = Trainer(model, optimizer, writer, logger, FOLD, EXPATH, EXNAME)
    trainer.fit(tdl, vdl, NUM_EPOCHS, start_epoch)
    trainer.submission(dl_test, submission_df)
    logger.close()


if __name__ == "__main__":
    main()
