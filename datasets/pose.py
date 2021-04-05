import json
import math
from typing import List, Tuple

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
from torch.utils.data import DataLoader, Dataset, Subset

from .common import HorizontalFlipEx, VerticalFlipEx


def reratio_box(roi: Tuple[float, float, float, float], ratio_limit=2.0):
    # 극단적인 비율의 이미지는 조정해줄 필요가 있음
    w, h = roi[2] - roi[0], roi[3] - roi[1]
    dhl, dhr, dwl, dwr = 0, 0, 0, 0
    if w / h > ratio_limit:
        # w/(h+x) = l, x = w/l - h
        dh = w / ratio_limit - h
        dhl, dhr = math.floor(dh / 2), math.ceil(dh / 2)
    elif h / w > ratio_limit:
        # h/(w+x) = l, x = h/l - w
        dw = h / ratio_limit - w
        dwl, dwr = math.floor(dw / 2), math.ceil(dw / 2)
    return dhl, dhr, dwl, dwr


class KeypointDataset(Dataset):
    def __init__(self, config, files, keypoints, augmentation):
        super().__init__()
        self.C = config
        self.files = files
        self.keypoints = keypoints

        T = []
        # T.append(A.Crop(0, 28, 1920, 1080 - 28))  # 1920x1080 --> 1920x1024
        # T.append(A.Resize(512, 1024))
        if augmentation:
            # 중간에 기구로 잘리는 경우를 가장
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
            T.append(A.OneOf(T_))

            # geomatric augmentations
            # T.append(A.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT))
            T.append(A.ShiftScaleRotate())
            T.append(HorizontalFlipEx())
            T.append(VerticalFlipEx())
            T.append(A.RandomRotate90())

            T_ = []
            T_.append(A.RandomBrightnessContrast(p=1))
            T_.append(A.RandomGamma(p=1))
            T_.append(A.RandomBrightness(p=1))
            T_.append(A.RandomContrast(p=1))
            T.append(A.OneOf(T_))

            T_ = []
            T_.append(A.MotionBlur(p=1))
            T_.append(A.GaussNoise(p=1))
            T.append(A.OneOf(T_))
        if self.C.dataset.normalize:
            if self.C.dataset.mean is not None and self.C.dataset.std is not None:
                T.append(A.Normalize(self.C.dataset.mean, self.C.dataset.std))
            else:
                T.append(A.Normalize())
        else:
            T.append(A.Normalize((0, 0, 0), (1, 1, 1)))
        T.append(ToTensorV2())

        self.transform = A.Compose(
            transforms=T,
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # rotation/scale 등으로 keypoint가 화면 밖으로 나가면 exception 발생.
        # 그럼 데이터 다시 만들어줌
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
                # if not self.C.dataset.normalize:
                #     image = image.type(torch.float) / 255.0
                bbox = list(map(int, a["bboxes"][0]))
                keypoint = torch.tensor(a["keypoints"], dtype=torch.float32)
                image, keypoint, heatmap, ratio, offset = self._resize_image(image, bbox, keypoint)

                return file, image, heatmap, ratio, offset
            except IndexError:
                pass

    def _resize_image(self, image, bbox, keypoint):
        """
        bbox크기만큼 이미지를 자르고, keypoint에 offset/ratio를 준다.
        """
        dhl, dhr, dwl, dwr = reratio_box(bbox, ratio_limit=self.C.dataset.ratio_limit)
        h, w = image.shape[1:3]
        bbox[0] = max(bbox[0] - dwl, 0)
        bbox[1] = max(bbox[1] - dhl, 0)
        bbox[2] = min(bbox[2] + dwr, w)
        bbox[3] = min(bbox[3] + dhr, h)

        image = image[:, bbox[1] : bbox[3], bbox[0] : bbox[2]]
        CD = self.C.dataset

        # HRNet의 입력 이미지 크기로 resize
        ratio = (CD.input_width / image.shape[2], CD.input_height / image.shape[1])
        ratio = torch.tensor(ratio, dtype=torch.float32)
        image = F.interpolate(image.unsqueeze(0), (CD.input_height, CD.input_width))[0]

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
        heatmap = utils.keypoints2heatmaps(
            keypoint,
            CD.input_height // 4,
            CD.input_width // 4,
            smooth=self.C.dataset.smooth_heatmap.do,
            smooth_size=self.C.dataset.smooth_heatmap.size,
            smooth_values=self.C.dataset.smooth_heatmap.values,
        )

        offset = torch.tensor([bbox[0], bbox[1]], dtype=torch.float)

        return image, keypoint, heatmap, ratio, offset


class TestKeypointDataset(Dataset):
    def __init__(
        self,
        files,
        info,
        normalize,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        size=(576, 768),
        rotation=0,
        horizontal_flip=False,
        vertical_flip=False,
        ratio_limit=2.0,
    ):
        super().__init__()
        self.files = files
        self.info = info
        self.size = size
        self.rotation = rotation % 4
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.ratio_limit = ratio_limit

        T = []
        if normalize:
            T.append(A.Normalize(mean, std))
        T.append(ToTensorV2())

        self.transform = A.Compose(transforms=T)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = str(self.files[idx])
        img = imageio.imread(file)
        # roi = self.info[idx]["roi"]
        roi = self.info[idx]

        # 극단적인 비율의 이미지는 조정해줄 필요가 있음
        h, w = img.shape[:2]
        roi = list(map(int, roi))
        dhl, dhr, dwl, dwr = reratio_box(roi, ratio_limit=self.ratio_limit)
        roi[0] = max(roi[0] - dwl, 0)
        roi[1] = max(roi[1] - dhl, 0)
        roi[2] = min(roi[2] + dwr, w)
        roi[3] = min(roi[3] + dhr, h)
        img = img[roi[1] : roi[3], roi[0] : roi[2]]
        offset = roi[:2]

        img = self.transform(image=img)["image"]

        if self.rotation > 0:
            img = torch.rot90(img, self.rotation, (1, 2))
        ori_size = (img.size(2), img.size(1))

        ratio = (self.size[0] / img.shape[2], self.size[1] / img.shape[1])
        img = F.interpolate(img.unsqueeze(0), self.size[::-1], mode="bilinear", align_corners=True).squeeze(0)

        if self.vertical_flip:
            img = torch.flip(img, (1,))
        if self.horizontal_flip:
            img = torch.flip(img, (2,))

        return file, img, offset, ratio, ori_size


def get_pose_datasets(C, fold):
    total_imgs = np.array(sorted(list(C.dataset.train_dir.glob("*.jpg"))))
    df = pd.read_csv(C.dataset.target_file)
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
    ds_train = KeypointDataset(
        C,
        total_imgs[train_idx],
        total_keypoints[train_idx],
        augmentation=True,
    )
    ds_valid = KeypointDataset(
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
