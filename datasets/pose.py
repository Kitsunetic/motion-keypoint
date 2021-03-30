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


class KeypointDataset(Dataset):
    def __init__(self, C, files, keypoints, augmentation):
        super().__init__()
        self.C = C
        self.files = files
        self.keypoints = keypoints

        T = []
        # T.append(A.Crop(0, 28, 1920, 1080 - 28))  # 1920x1080 --> 1920x1024
        # T.append(A.Resize(512, 1024))
        if augmentation:
            T.append(A.ImageCompression())
            T.append(A.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, rotate_limit=30))
            T.append(HorizontalFlipEx())
            T.append(VerticalFlipEx())
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
        while True:
            try:
                file = str(self.files[idx])
                image = imageio.imread(file)

                keypoint = self.keypoints[idx]
                box = utils.keypoint2box(keypoint, self.C.dataset.padding)
                box = np.expand_dims(box, 0)
                labels = np.array([0], dtype=np.int64)
                a = self.transform(image=image, labels=labels, bboxes=box, keypoints=keypoint)

                image = a["image"]
                bbox = list(map(int, a["bboxes"][0]))
                keypoint = torch.tensor(a["keypoints"], dtype=torch.float32)
                image, keypoint, heatmap, ratio = self._resize_image(image, bbox, keypoint)

                return file, image, keypoint, heatmap, ratio
            except IndexError:
                pass

    def _resize_image(self, image, bbox, keypoint):
        """
        bbox크기만큼 이미지를 자르고, keypoint에 offset/ratio를 준다.
        """
        image = image[:, bbox[1] : bbox[3], bbox[0] : bbox[2]]
        cfg = self.C.dataset

        # HRNet의 입력 이미지 크기로 resize
        ratio = (cfg.input_width / image.shape[2], cfg.input_height / image.shape[1])
        ratio = torch.tensor(ratio, dtype=torch.float32)
        image = F.interpolate(image.unsqueeze(0), (cfg.input_height, cfg.input_width))[0]

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
        heatmap = utils.keypoints2heatmaps(keypoint, cfg.input_height // 4, cfg.input_width // 4)

        return image, keypoint, heatmap, ratio


class TestKeypointDataset(Dataset):
    def __init__(self, files, offsets, ratios, augmentation):
        super().__init__()
        self.files = files
        self.offsets = offsets
        self.ratios = ratios

        T = []
        if augmentation:
            T.append(A.ImageCompression())
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


def get_pose_datasets(config, fold):
    total_imgs = np.array(sorted(list((config.dataset.dir / "train_imgs").glob("*.jpg"))))
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
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.seed)
    indices = list(skf.split(total_imgs, groups))
    train_idx, valid_idx = indices[fold - 1]

    # 데이터셋 생성
    ds_train = KeypointDataset(
        config,
        total_imgs[train_idx],
        total_keypoints[train_idx],
        augmentation=True,
    )
    ds_valid = KeypointDataset(
        config,
        total_imgs[valid_idx],
        total_keypoints[valid_idx],
        augmentation=False,
    )
    dl_train = DataLoader(
        ds_train,
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_cpus,
        shuffle=True,
        pin_memory=True,
    )
    dl_valid = DataLoader(
        ds_valid,
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_cpus,
        shuffle=False,
        pin_memory=True,
    )

    # 테스트 데이터셋
    test_files = sorted(list((config.dataset.dir / "test_imgs").glob("*.jpg")))
    with open("data/test_imgs_effdet/data.json", "r") as f:
        data = json.load(f)
        offsets = data["offset"]
        ratios = data["ratio"]
    ds_test = TestKeypointDataset(
        test_files,
        offsets,
        ratios,
        augmentation=False,
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_cpus,
        shuffle=False,
        pin_memory=True,
    )

    return dl_train, dl_valid, dl_test
