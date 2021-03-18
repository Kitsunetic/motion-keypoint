import math
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.augmentations import functional as AF
from albumentations.core.transforms_interface import DualTransform
from torch.utils.data import Dataset


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def generate_experiment_directory(base_dir, comments=None):
    comments = list(map(lambda x: x.replace(" ", "_"), filter(None, comments)))

    base_dir = Path(base_dir)
    n = datetime.now()
    dirname = f"{n.month:02d}{n.day:02d}_{n.hour:02d}{n.minute:02d}{n.second:02d}"
    if isinstance(comments, str):
        dirname += f"-{comments}"
    elif isinstance(comments, Iterable):
        dirname += "-" + "-".join(comments)
    dirpath = base_dir / dirname
    (dirpath / "example").mkdir(parents=True, exist_ok=True)

    return dirpath, dirpath.name


class AverageMeter(object):
    """
    AverageMeter, referenced to https://dacon.io/competitions/official/235626/codeshare/1684
    """

    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.losses = []

    def update(self, val, n=1):
        self.losses.append(val)
        if n > 0:
            self.sum += val * n
            self.cnt += n
            self.avg = self.sum / self.cnt

    def get(self):
        return self.avg

    def __call__(self):
        return self.avg


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0
        ].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack(
                [
                    p.grad.norm(p=2).to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm


class CustomLogger:
    def __init__(self, filename, filemode="a", use_color=True):
        filename = Path(filename)
        if filename.is_dir():
            timestr = self._get_timestr().replace(" ", "_").replace(":", "-")
            filename = filename / f"log_{timestr}.log"
        self.file = open(filename, filemode)
        self.use_color = use_color

    def _get_timestr(self):
        n = datetime.now()
        return f"{n.year:04d}-{n.month:02d}-{n.day:02d} {n.hour:02d}:{n.minute:02d}:{n.second:02d}"

    def _write(self, msg, level):
        timestr = self._get_timestr()
        out = f"[{timestr} {level}] {msg}"

        if self.use_color:
            if level == " INFO":
                print("\033[34m" + out + "\033[0m")
            elif level == " WARN":
                print("\033[35m" + out + "\033[0m")
            elif level == "ERROR":
                print("\033[31m" + out + "\033[0m")
            elif level == "FATAL":
                print("\033[43m\033[1m" + out + "\033[0m")
            else:
                print(out)
        else:
            print(out)
        self.file.write(out + "\r\n")

    def debug(self, *msg):
        msg = " ".join(map(str, msg))
        self._write(msg, "DEBUG")

    def info(self, *msg):
        msg = " ".join(map(str, msg))
        self._write(msg, " INFO")

    def warn(self, *msg):
        msg = " ".join(map(str, msg))
        self._write(msg, " WARN")

    def error(self, *msg):
        msg = " ".join(map(str, msg))
        self._write(msg, "ERROR")

    def fatal(self, *msg):
        msg = " ".join(map(str, msg))
        self._write(msg, "FATAL")

    def flush(self):
        self.file.flush()


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


def draw_keypoints(image: np.ndarray, keypoints: np.ndarray):
    edges = [
        (0, 1),
        (0, 2),
        (2, 4),
        (1, 3),
        (6, 8),
        (8, 10),
        (5, 7),
        (7, 9),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
        (5, 6),
        (15, 22),
        (16, 23),
        (11, 21),
        (21, 12),
        (20, 21),
        (5, 20),
        (6, 20),
        (17, 6),
        (17, 5),
    ]
    keypoint_names = [
        "nose_x",
        "nose_y",
        "left_eye_x",
        "left_eye_y",
        "right_eye_x",
        "right_eye_y",
        "left_ear_x",
        "left_ear_y",
        "right_ear_x",
        "right_ear_y",
        "left_shoulder_x",
        "left_shoulder_y",
        "right_shoulder_x",
        "right_shoulder_y",
        "left_elbow_x",
        "left_elbow_y",
        "right_elbow_x",
        "right_elbow_y",
        "left_wrist_x",
        "left_wrist_y",
        "right_wrist_x",
        "right_wrist_y",
        "left_hip_x",
        "left_hip_y",
        "right_hip_x",
        "right_hip_y",
        "left_knee_x",
        "left_knee_y",
        "right_knee_x",
        "right_knee_y",
        "left_ankle_x",
        "left_ankle_y",
        "right_ankle_x",
        "right_ankle_y",
        "neck_x",
        "neck_y",
        "left_palm_x",
        "left_palm_y",
        "right_palm_x",
        "right_palm_y",
        "spine2(back)_x",
        "spine2(back)_y",
        "spine1(waist)_x",
        "spine1(waist)_y",
        "left_instep_x",
        "left_instep_y",
        "right_instep_x",
        "right_instep_y",
    ]
    image = image.copy()

    np.random.seed(42)
    colors = {k: tuple(map(int, np.random.randint(0, 255, 3))) for k in range(24)}
    x1, y1 = max(0, min(keypoints[:, 0]) - 10), max(0, min(keypoints[:, 1]) - 10)
    x2, y2 = min(image.shape[1], max(keypoints[:, 0]) + 10), min(image.shape[0], max(keypoints[:, 1]) + 10)
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 100, 91), 3)

    for i, keypoint in enumerate(keypoints):
        cv2.circle(image, tuple(keypoint), 3, colors.get(i), thickness=3, lineType=cv2.FILLED)

        cv2.putText(
            image,
            f"{i}: {keypoint_names[i*2]}",
            tuple(keypoint),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )

    for i, edge in enumerate(edges):
        cv2.line(
            image,
            tuple(keypoints[edge[0]]),
            tuple(keypoints[edge[1]]),
            colors.get(edge[0]),
            3,
            lineType=cv2.LINE_AA,
        )

    return image


def draw_keypoints_show(image: np.ndarray, keypoints: np.ndarray):
    image = draw_keypoints(image, keypoints)

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.tight_layout()
    # plt.savefig("example.png")
    # imageio.imwrite("example.png", image)
    plt.show()


def channels2keypoints(p: torch.Tensor):
    # input: [24, 192, 144]
    pos = torch.argmax(p.flatten(1), 1)
    y = pos // 144
    x = pos % 144
    return torch.stack([x, y], 1)


def keypoints2channels(k: torch.Tensor, h=768 // 4, w=576 // 4):
    k = k.type(torch.int64)
    c = torch.zeros(k.size(0), h, w, dtype=torch.float32)
    for i, (x, y) in enumerate(k):
        c[i, y, x] = 1.0
    return c


def nums2keypoints(nums):
    x = nums % 144
    y = nums // 144
    return torch.stack([x, y], 1)


def get_single_person_rois(out):
    max_area = 0
    max_idx = 0
    for i, (roi, class_id, score) in enumerate(zip(out["rois"], out["class_ids"], out["scores"])):
        if class_id != 0:
            continue

        area = (roi[2] - roi[0]) * (roi[3] - roi[1])
        if max_area < area:
            max_area = area
            max_idx = i

    return out["rois"][max_idx].astype(np.int64)


def resize_box(
    img: np.ndarray,
    opad: int,
    box: List[int],
    keypoint: np.ndarray = None,
    dst_h=768,
    dst_w=576,
    ori_pad_h=38,
    ori_pad_w=374,
):
    """
    - img: 입력 이미지
    - keypoint: 입력 키포인트
    - opad: 키포인트 바깥에 기본적으로 둘 padding
    - box: detection 모델에서 구해진 roi
    - dst_h, dst_w: 최종 이미지의 사이즈
    - ori_pad_h, ori_pad_w: 기본적으로 더해지는 offset
    """
    x = img

    box = [box[0] - opad, box[1] - opad, box[2] + opad, box[3] + opad]
    h, w = box[3] - box[1], box[2] - box[0]
    if h > w:
        ratio = dst_h / h
        jw = dst_w / ratio
        center_w = (box[0] + box[2]) / 2
        dbox = [int(center_w - jw / 2), box[1], int(center_w + jw / 2), box[3]]

        # 초과분은 zero padding
        zpad = [0, 0]
        if dbox[0] < 0:
            zpad[0] = -dbox[0]
            dbox[0] = 0
        if dbox[2] > x.shape[1]:
            zpad[1] = dbox[2] - x.shape[1]
            dbox[2] = x.shape[1]

        x = x[dbox[1] : dbox[3], dbox[0] : dbox[2]]
        pad1 = np.zeros((x.shape[0], zpad[0], 3), dtype=np.uint8)
        pad2 = np.zeros((x.shape[0], zpad[1], 3), dtype=np.uint8)
        x = np.concatenate([pad1, x, pad2], 1)
    else:
        ratio = dst_w / w
        jh = dst_h / ratio
        center_h = (box[1] + box[3]) / 2
        dbox = [box[0], int(center_h - jh / 2), box[2], int(center_h + jh / 2)]

        # 초과분은 zero padding
        zpad = [0, 0]
        if dbox[1] < 0:
            zpad[0] = -dbox[1]
            dbox[1] = 0
        if dbox[3] > x.shape[0]:
            zpad[1] = dbox[3] - x.shape[0]
            dbox[3] = x.shape[0]

        x = x[dbox[1] : dbox[3], dbox[0] : dbox[2]]
        pad1 = np.zeros((zpad[0], x.shape[1], 3), dtype=np.uint8)
        pad2 = np.zeros((zpad[1], x.shape[1], 3), dtype=np.uint8)
        x = np.concatenate([pad1, x, pad2], 0)

    x = cv2.resize(x, (dst_w, dst_h), interpolation=cv2.INTER_LANCZOS4)
    offset = [int(dbox[0] + ori_pad_w), int(dbox[1] + ori_pad_h)]

    if keypoint is not None:
        k = keypoint.copy()
        k[:, 0] -= offset[0]
        k[:, 1] -= offset[1]
        k *= ratio

        return x, k, offset, ratio
    else:
        return x, offset, ratio


def imshow_horizon(*ims, figsize=(12, 6)):
    plt.figure(figsize=figsize)
    for i, im in enumerate(ims, 1):
        plt.subplot(1, len(ims), i)
        plt.imshow(im)
    plt.tight_layout()
    plt.show()


class HorizontalFlipEx(DualTransform):
    """Flip the input horizontally around the y-axis.
    Args:
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image, mask, bboxes, keypoints
    Image types:
        uint8, float32
    """

    def __init__(self, always_apply, p):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        if img.ndim == 3 and img.shape[2] > 1 and img.dtype == np.uint8:
            # Opencv is faster than numpy only in case of
            # non-gray scale 8bits images
            return AF.hflip_cv2(img)

        return AF.hflip(img)

    def apply_to_bbox(self, bbox, **params):
        return AF.bbox_hflip(bbox, **params)

    def apply_to_keypoint(self, keypoint, **params):
        keypoint = AF.keypoint_hflip(keypoint, **params)
        # TODO left/right 키포인트들은 서로 swap해주기
        return keypoint

    def get_transform_init_args_names(self):
        return ()
