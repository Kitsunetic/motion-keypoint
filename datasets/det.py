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
        self.config = config
        self.files = files
        self.keypoints = keypoints

        T = []
        T.append(A.Crop(*config.crop))
        T.append(A.Resize(config.input_height, config.input_width))
        if augmentation:
            # cutout
            T_ = []
            T_.append(A.Cutout(max_h_size=20, max_w_size=20))
            T_.append(A.Cutout(max_h_size=20, max_w_size=20, fill_value=255))
            T_.append(A.Cutout(max_h_size=config.input_height // 2, max_w_size=10, fill_value=255))
            T_.append(A.Cutout(max_h_size=config.input_height // 2, max_w_size=10, fill_value=0))
            T_.append(A.Cutout(max_h_size=10, max_w_size=config.input_width // 2, fill_value=255))
            T_.append(A.Cutout(max_h_size=10, max_w_size=config.input_width // 2, fill_value=0))
            T.append(A.OneOf(T_))

            # geometric
            T.append(A.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT))
            T.append(HorizontalFlipEx())
            # T.append(VerticalFlipEx())
            # T.append(A.RandomRotate90())

            # impressive
            # T.append(A.ImageCompression())
            T.append(A.IAASharpen())  # 이거 뭔지?
            T_ = []
            T_.append(A.RandomBrightnessContrast())
            T_.append(A.RandomGamma())
            T_.append(A.RandomBrightness())
            T_.append(A.RandomContrast())
            T.append(A.OneOf(T_))
            # T.append(A.GaussNoise())
            T.append(A.Blur())
        T.append(A.Normalize())
        T.append(ToTensorV2())

        self.transform = A.Compose(
            transforms=T,
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
            # keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
            # TODO 영역을 벗어난 keypoint는 그 영역의 한도 값으로 설정해줄 것?
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = str(self.files[idx])
        image = imageio.imread(file)

        keypoint = self.keypoints[idx]
        box = utils.keypoint2box(keypoint, self.config.padding)
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

        T = []
        T.append(A.Crop(*config.crop))
        T.append(A.Resize(config.input_height, config.input_width))
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


def get_det_dataset(config, fold):
    total_imgs = np.array(sorted(list((config.data_dir / "train_imgs").glob("*.jpg"))))
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

    # KFold
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=config.seed)
    indices = list(skf.split(total_imgs, groups))
    train_idx, valid_idx = indices[fold - 1]

    # 데이터셋 생성
    ds_train = DetDataset(
        config,
        total_imgs[train_idx],
        total_keypoints[train_idx],
        augmentation=True,
    )
    ds_valid = DetDataset(
        config,
        total_imgs[valid_idx],
        total_keypoints[valid_idx],
        augmentation=False,
    )
    dl_train = DataLoader(
        ds_train,
        batch_size=config.batch_size,
        num_workers=config.num_cpus,
        shuffle=True,
        pin_memory=True,
    )
    dl_valid = DataLoader(
        ds_valid,
        batch_size=config.batch_size,
        num_workers=config.num_cpus,
        shuffle=False,
        pin_memory=True,
    )

    # 테스트 데이터셋
    test_files = sorted(list((config.data_dir / "test_imgs").glob("*.jpg")), reverse=True)
    ds_test = TestDetDataset(
        config,
        test_files,
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=config.batch_size,
        num_workers=config.num_cpus,
        shuffle=False,
        pin_memory=True,
    )

    return dl_train, dl_valid, dl_test
