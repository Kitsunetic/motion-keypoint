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
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.model_selection import KFold, StratifiedKFold
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Subset

import utils
from error_list import error_list


class HorizontalFlipEx(A.HorizontalFlip):
    swap_columns = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (18, 19), (22, 23)]

    def apply_to_keypoints(self, keypoints, **params):
        keypoints = super().apply_to_keypoints(keypoints, **params)

        # left/right 키포인트들은 서로 swap해주기
        for a, b in self.swap_columns:
            temp1 = deepcopy(keypoints[a])
            temp2 = deepcopy(keypoints[b])
            keypoints[a] = temp2
            keypoints[b] = temp1

        return keypoints


class VerticalFlipEx(A.VerticalFlip):
    swap_columns = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (18, 19), (22, 23)]

    def apply_to_keypoints(self, keypoints, **params):
        keypoints = super().apply_to_keypoints(keypoints, **params)

        # left/right 키포인트들은 서로 swap해주기
        for a, b in self.swap_columns:
            temp1 = deepcopy(keypoints[a])
            temp2 = deepcopy(keypoints[b])
            keypoints[a] = temp2
            keypoints[b] = temp1

        return keypoints


class KeypointDataset(Dataset):
    def __init__(self, config, files, keypoints, augmentation, padding):
        super().__init__()
        self.config = config
        self.files = files
        self.keypoints = keypoints
        self.padding = padding

        T = []
        # T.append(A.Crop(0, 28, 1920, 1080 - 28))  # 1920x1080 --> 1920x1024
        # T.append(A.Resize(512, 1024))
        if augmentation:
            T.append(A.ImageCompression())
            T.append(A.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT))
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
        ratio = (self.config.input_width / image.shape[2], self.config.input_height / image.shape[1])
        ratio = torch.tensor(ratio, dtype=torch.float32)
        image = F.interpolate(image.unsqueeze(0), (self.config.input_height, self.config.input_width))[0]

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
        heatmap = utils.keypoints2heatmaps(keypoint, self.config.input_height // 4, self.config.input_width // 4)

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
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.seed)
    indices = list(skf.split(total_imgs, groups))
    train_idx, valid_idx = indices[fold - 1]

    # 데이터셋 생성
    ds_train = KeypointDataset(
        config,
        total_imgs[train_idx],
        total_keypoints[train_idx],
        augmentation=True,
        padding=config.padding,
    )
    ds_valid = KeypointDataset(
        config,
        total_imgs[valid_idx],
        total_keypoints[valid_idx],
        augmentation=False,
        padding=config.padding,
    )
    dl_train = DataLoader(ds_train, batch_size=config.batch_size, shuffle=True, pin_memory=True)
    dl_valid = DataLoader(ds_valid, batch_size=config.batch_size, shuffle=False, pin_memory=True)

    # 테스트 데이터셋
    test_files = sorted(list((config.data_dir / "test_imgs").glob("*.jpg")))
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
    dl_test = DataLoader(ds_test, batch_size=config.batch_size, shuffle=False, pin_memory=True)

    return dl_train, dl_valid, dl_test
