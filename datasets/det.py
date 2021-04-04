import json
from copy import deepcopy
from pathlib import Path

import albumentations as A
import cv2
import imageio
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import utils
from albumentations.pytorch import ToTensorV2
from error_list import error_list
from PIL import Image
from sklearn.model_selection import KFold, StratifiedKFold
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Subset

from .common import HorizontalFlipEx, VerticalFlipEx


class DetDataset(Dataset):
    def __init__(self, config, files, keypoints, augmentation):
        super().__init__()
        self.C = config
        self.files = files
        self.keypoints = keypoints

        T = []
        T.append(A.Crop(*self.C.dataset.crop))
        T.append(A.Resize(self.C.dataset.input_height, self.C.dataset.input_width))
        if augmentation:
            T_ = []
            T_.append(A.Cutout(num_holes=16, max_h_size=100, max_w_size=100, fill_value=0, p=1))
            T_.append(A.Cutout(num_holes=16, max_h_size=100, max_w_size=100, fill_value=255, p=1))
            T_.append(A.Cutout(num_holes=16, max_h_size=100, max_w_size=100, fill_value=128, p=1))
            T_.append(A.Cutout(num_holes=16, max_h_size=100, max_w_size=100, fill_value=192, p=1))
            T_.append(A.Cutout(num_holes=16, max_h_size=100, max_w_size=100, fill_value=64, p=1))
            T_.append(A.Cutout(num_holes=5, max_h_size=1920, max_w_size=50, fill_value=0, p=1))
            T_.append(A.Cutout(num_holes=5, max_h_size=1920, max_w_size=50, fill_value=255, p=1))
            T_.append(A.Cutout(num_holes=5, max_h_size=1920, max_w_size=50, fill_value=128, p=1))
            T_.append(A.Cutout(num_holes=5, max_h_size=1920, max_w_size=50, fill_value=192, p=1))
            T_.append(A.Cutout(num_holes=5, max_h_size=1920, max_w_size=50, fill_value=64, p=1))
            T_.append(A.Cutout(num_holes=5, max_h_size=30, max_w_size=1080, fill_value=0, p=1))
            T_.append(A.Cutout(num_holes=5, max_h_size=30, max_w_size=1080, fill_value=255, p=1))
            T_.append(A.Cutout(num_holes=5, max_h_size=30, max_w_size=1080, fill_value=128, p=1))
            T_.append(A.Cutout(num_holes=5, max_h_size=30, max_w_size=1080, fill_value=192, p=1))
            T_.append(A.Cutout(num_holes=5, max_h_size=30, max_w_size=1080, fill_value=64, p=1))
            # T_.append(A.Cutout(max_h_size=20, max_w_size=20))
            # T_.append(A.Cutout(max_h_size=20, max_w_size=20, fill_value=255))
            # T_.append(A.Cutout(max_h_size=self.C.dataset.input_height // 2, max_w_size=10, fill_value=255))
            # T_.append(A.Cutout(max_h_size=self.C.dataset.input_height // 2, max_w_size=10, fill_value=0))
            # T_.append(A.Cutout(max_h_size=10, max_w_size=self.C.dataset.input_width // 2, fill_value=255))
            # T_.append(A.Cutout(max_h_size=10, max_w_size=self.C.dataset.input_width // 2, fill_value=0))
            T.append(A.OneOf(T_))

            T.append(A.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT))
            T.append(HorizontalFlipEx())
            T.append(VerticalFlipEx())
            # T.append(A.RandomRotate90()) # batch-augmentation으로 대체

            T_ = []
            T_.append(A.RandomBrightnessContrast())
            T_.append(A.RandomGamma())
            T_.append(A.RandomBrightness())
            T_.append(A.RandomContrast())
            T.append(A.OneOf(T_))

            T_ = []
            T_.append(A.MotionBlur(p=1))
            T_.append(A.GaussNoise(p=1))
            T.append(A.OneOf(T_))
        T.append(A.Normalize())
        T.append(ToTensorV2())

        self.transform = A.Compose(
            transforms=T,
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
            # keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = str(self.files[idx])
        image = imageio.imread(file)

        keypoint = self.keypoints[idx]
        box = utils.keypoint2box(keypoint, self.C.dataset.padding)
        box = np.expand_dims(box, 0)
        labels = np.array([0], dtype=np.int64)
        a = self.transform(image=image, labels=labels, bboxes=box)

        image = a["image"]

        annot = np.zeros((1, 5), dtype=np.float32)
        annot[0, :4] = a["bboxes"][0]
        annot = torch.tensor(annot, dtype=torch.float32)

        return file, image, annot


class TestDetDataset(Dataset):
    def __init__(self, config, files):
        super().__init__()
        self.config = config
        self.files = files
        dataset = self.C.dataset

        T = []
        T.append(A.Crop(*dataset.crop))
        T.append(A.Resize(dataset.input_height, dataset.input_width))
        T.append(A.Normalize())
        T.append(ToTensorV2())

        self.transform = A.Compose(transforms=T)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = str(self.files[idx])
        image = imageio.imread(file)
        a = self.transform(image=image)

        image = a["image"]

        return file, image


def get_det_dataset(C, fold):
    datadir = Path(C.dataset.dir)
    total_imgs = np.array(sorted(list((datadir / "train_imgs").glob("*.jpg"))))
    df = pd.read_csv(datadir / "train_df.csv")
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

    # KFold
    if C.dataset.group_kfold:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=C.seed)
        # 파일 이름 앞 17자리를 group으로 이미지를 분류 (파일이 너무 잘 섞여도 안됨)
        groups = []
        last_group = 0
        last_stem = total_imgs[0].name[:17]
        for f in total_imgs:
            stem = f.name[:17]
            if stem == last_stem:
                groups.append(last_group)
            else:
                last_group += 1
                last_stem = stem
                groups.append(last_group)
        indices = list(skf.split(total_imgs, groups))
    else:
        kf = KFold(n_splits=5, shuffle=True, random_state=C.seed)
        indices = list(kf.split(total_imgs))
    train_idx, valid_idx = indices[fold - 1]

    # 데이터셋 생성
    ds_train = DetDataset(
        C,
        total_imgs[train_idx],
        total_keypoints[train_idx],
        augmentation=True,
    )
    ds_valid = DetDataset(
        C,
        total_imgs[valid_idx],
        total_keypoints[valid_idx],
        augmentation=False,
    )
    dl_train = DataLoader(
        ds_train,
        batch_size=C.dataset.batch_size,
        num_workers=C.dataset.num_cpus,
        shuffle=True,
        pin_memory=True,
    )
    dl_valid = DataLoader(
        ds_valid,
        batch_size=C.dataset.batch_size,
        num_workers=C.dataset.num_cpus,
        shuffle=False,
        pin_memory=True,
    )

    return dl_train, dl_valid
