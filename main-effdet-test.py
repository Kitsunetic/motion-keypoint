import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from easydict import EasyDict
from PIL import Image
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

import datasets
import networks
import options
import utils
from datasets import get_det_dataset


class DetTester:
    def __init__(self, C):
        self.C = C

        self.det_model = networks.EfficientDet(self.C.det_model.name, pretrained=True)
        self.det_model.cuda()
        if self.C.det_model.pretrained is not None:
            print("Load", self.C.det_model.pretrained)
            self.det_model.load_state_dict(torch.load(self.C.det_model.pretrained)["model"])

    def get_dataloader(self, dpath):
        dpath = Path(dpath)
        files = sorted(list(dpath.glob("*.jpg")))
        ds = datasets.TestDetDataset(self.C, files)
        dl = DataLoader(
            ds,
            batch_size=self.C.dataset.batch_size,
            shuffle=False,
            num_workers=self.C.dataset.num_cpus,
            pin_memory=True,
        )
        return dl

    @staticmethod
    def _summary_results(elems, dw, dh, ratio_x, ratio_y, fliplr=False, n_rotate=0):
        rois, scores = [], []
        for elem in elems:
            class_mask = elem["class_ids"] == 0
            if len(elem["rois"][class_mask]) > 0:
                k = np.argmax(elem["scores"][class_mask])
                roi = elem["rois"][class_mask][k]
                score = elem["scores"][class_mask][k]

                n_rotate %= 4
                if n_rotate == 1:
                    a, b, c, d = roi
                    roi = np.array([dw - d, a, dw - b, c])
                elif n_rotate == 2:
                    a, b, c, d = roi
                    roi = np.array([dw - c, dh - d, dw - a, dh - b])
                elif n_rotate == 3:
                    a, b, c, d = roi
                    roi = np.array([b, dh - c, d, dh - a])

                if fliplr:
                    roi[0::2] = (dw - roi[0::2]) / ratio_x
                    roi[[0, 2]] = roi[[2, 0]]
                    roi[1::2] /= ratio_y
                else:
                    roi[0::2] /= ratio_x
                    roi[1::2] /= ratio_y

                rois.append(roi)
                scores.append(score)
            else:
                rois.append(None)
                scores.append(None)

        return rois, scores

    @torch.no_grad()
    def evaluate(self, dl: DataLoader, file_out_dir):
        self.det_model.eval()
        file_out_dir = Path(file_out_dir)
        file_out_dir.mkdir(parents=True, exist_ok=True)
        test = self.C.test

        lst_rois = []
        with tqdm(total=len(dl.dataset), ncols=100, file=sys.stdout) as t:
            for files, imgs in dl:
                imgs_ = imgs.cuda(non_blocking=True)
                batch_size, _, h, w = imgs.shape

                # multiscale tests
                rois = []  # num_tests, batch_size, 4
                scores = []  # num_tests, batch_size
                for dw, dh in test.multiscale_test.sizes:
                    ratio_x, ratio_y = dw / w, dh / h
                    imgs_resize_ = F.interpolate(imgs_, (dh, dw))
                    elems = self.det_model(imgs_resize_)
                    roi, score = self._summary_results(elems, dw, dh, ratio_x, ratio_y, fliplr=False)
                    rois.append(roi)
                    scores.append(score)

                    # horizontal flip test
                    if test.flip_test.horizontal:
                        imgs_flip_ = torch.flip(imgs_resize_, dims=[3])
                        elems = self.det_model(imgs_flip_)
                        roi, score = self._summary_results(elems, dw, dh, ratio_x, ratio_y, fliplr=True)
                        rois.append(roi)
                        scores.append(score)

                    print()
                    # rotate90 left test
                    if test.rotate_test.left:
                        imgs_rot_ = torch.rot90(imgs_resize_, k=1, dims=(2, 3))
                        elems = self.det_model(imgs_rot_)
                        roi, score = self._summary_results(elems, dw, dh, ratio_x, ratio_y, fliplr=False, n_rotate=1)
                        rois.append(roi)
                        scores.append(score)
                        if roi[0] is not None:
                            imgs_rot_np = imageio.imread(files[0])
                            print(roi[0], score[0], imgs_rot_np.shape)
                            roii = roi[0].astype(np.int)
                            cv2.rectangle(imgs_rot_np, (roii[0], roii[1]), (roii[2], roii[3]), (255, 0, 0), 1)
                            imageio.imwrite("test1.jpg", imgs_rot_np)
                    if test.rotate_test.left and test.flip_test.horizontal:
                        imgs_rot_ = torch.rot90(imgs_flip_, k=1, dims=(2, 3))
                        elems = self.det_model(imgs_rot_)
                        roi, score = self._summary_results(elems, dw, dh, ratio_x, ratio_y, fliplr=True, n_rotate=1)
                        rois.append(roi)
                        scores.append(score)
                        if roi[0] is not None:
                            imgs_rot_np = imageio.imread(files[0])
                            print(roi[0], score[0], imgs_rot_np.shape)
                            roii = roi[0].astype(np.int)
                            cv2.rectangle(imgs_rot_np, (roii[0], roii[1]), (roii[2], roii[3]), (255, 0, 0), 1)
                            imageio.imwrite("test2.jpg", imgs_rot_np)

                    # rotate90 right test
                    if test.rotate_test.right:
                        imgs_rot_ = torch.rot90(imgs_resize_, k=3, dims=(2, 3))
                        elems = self.det_model(imgs_rot_)
                        roi, score = self._summary_results(elems, dw, dh, ratio_x, ratio_y, fliplr=False, n_rotate=3)
                        rois.append(roi)
                        scores.append(score)
                        if roi[0] is not None:
                            imgs_rot_np = imageio.imread(files[0])
                            print(roi[0], score[0], imgs_rot_np.shape)
                            roii = roi[0].astype(np.int)
                            cv2.rectangle(imgs_rot_np, (roii[0], roii[1]), (roii[2], roii[3]), (255, 0, 0), 1)
                            imageio.imwrite("test3.jpg", imgs_rot_np)
                    if test.rotate_test.right and test.flip_test.horizontal:
                        imgs_rot_ = torch.rot90(imgs_flip_, k=3, dims=(2, 3))
                        elems = self.det_model(imgs_rot_)
                        roi, score = self._summary_results(elems, dw, dh, ratio_x, ratio_y, fliplr=True, n_rotate=3)
                        rois.append(roi)
                        scores.append(score)
                        if roi[0] is not None:
                            imgs_rot_np = imageio.imread(files[0])
                            print(roi[0], score[0], imgs_rot_np.shape)
                            roii = roi[0].astype(np.int)
                            cv2.rectangle(imgs_rot_np, (roii[0], roii[1]), (roii[2], roii[3]), (255, 0, 0), 1)
                            imageio.imwrite("test4.jpg", imgs_rot_np)

                # summarize rois
                out_rois = []
                for i in range(batch_size):
                    batch_rois, batch_scores = [], []
                    for j in range(len(rois)):
                        if rois[j][i] is not None:
                            batch_rois.append(rois[j][i])
                            batch_scores.append(scores[j][i])

                    roi = np.stack(batch_rois)
                    score = np.stack(batch_scores)
                    if self.C.test.voting_method == "mean":
                        roi = np.average(roi, axis=0)
                    elif self.C.test.voting_method == "weighted_mean":
                        roi = np.average(roi, axis=0, weights=score)
                    elif self.C.test.voting_method == "median":
                        roi = np.median(roi, axis=0)
                    out_rois.append(roi)

                # save as image
                for file, roi in zip(files, out_rois):
                    file = Path(file)
                    t.set_postfix_str(file.stem)

                    ud_roi = roi.copy()
                    ow, oh = self.C.dataset.input_width, self.C.dataset.input_height
                    cw, ch = self.C.dataset.crop[0], self.C.dataset.crop[1]
                    ud_roi[0::2] = ud_roi[0::2] / ow * (1920 - cw * 2) + cw
                    ud_roi[1::2] = ud_roi[1::2] / oh * (1080 - ch * 2) + ch

                    lst_roi = ud_roi.tolist()
                    lst_rois.append(dict(file=file.name, roi=lst_roi))

                    int_roi = ud_roi.astype(np.int)
                    img_np = imageio.imread(file)
                    clip = img_np[int_roi[1] : int_roi[3], int_roi[0] : int_roi[2]]
                    imageio.imwrite(file_out_dir / file.name, clip)

                    t.update()

        # save roi information as file
        with open(file_out_dir / "info.json", "w") as f:
            json.dump(lst_rois, f)

    def scale_inference_single_image(self, img, dw, dh, flip=False):
        _, h, w = img.shape
        img = img.unsqueeze(0)

        ratio_x = dw / w
        ratio_y = dh / h
        img = F.interpolate(img, (dh, dw))
        if flip:
            img = torch.flip(img, dims=[3])
        result = self.det_model(img)[0]

        class_mask = result["class_ids"] == 0
        rois = result["rois"][class_mask]
        scores = result["scores"][class_mask]

        if len(rois) > 0:
            j = np.argmax(scores)
            roi = rois[j]
            score = scores[j]

            if flip:
                roi[0::2] = (dw - roi[0::2]) / ratio_x
                roi[1::2] /= ratio_y
                temp = roi[0].copy()
                roi[0] = roi[2]
                roi[2] = temp
            else:
                roi[0::2] /= ratio_x
                roi[1::2] /= ratio_y

            return roi, score
        else:
            return None, None

    def multiscale_inference(self, imgs):
        """사이즈/LR에 대해 bbox 좌표를 구하고, voting이 아니라 좌표값에 대한 가중평균을 구함"""
        rectified_rois = []

        for img in imgs:
            rois = []
            scores = []
            for dw, dh in self.C.inference.multiscale_test.sizes:
                roi, score = self.scale_inference_single_image(img, dw, dh, flip=False)
                if roi is not None:
                    rois.append(roi)
                    scores.append(score)

                if self.C.inference.flip_test.horizontal:
                    roi, score = self.scale_inference_single_image(img, dw, dh, flip=True)
                    if roi is not None:
                        rois.append(roi)
                        scores.append(score)

            # print(len(rois) )

            rois = np.stack(rois)
            scores = np.stack(scores)
            new_roi = np.average(rois, axis=0, weights=scores)
            rectified_rois.append(new_roi)

        return np.stack(rectified_rois)

    @torch.no_grad()
    def test_loop(self, file_out_dir):
        self.det_model.eval()
        file_out_dir = Path(file_out_dir)
        file_out_dir.mkdir(parents=True, exist_ok=True)

        inf = self.C.inference

        with tqdm(total=len(self.dl_test.dataset), ncols=100, file=sys.stdout) as t:
            for files, imgs in self.dl_test:
                imgs_ = imgs.cuda(non_blocking=True)

                # multi-scale test
                pred_bboxes = self.multiscale_inference(imgs_)

                for file, pred_bbox in zip(files, pred_bboxes):
                    file = Path(file)
                    t.set_postfix_str(file.name)

                    ud_bbox = pred_bbox.copy()
                    ud_bbox[0::2] = ud_bbox[0::2] / self.C.input_width * (1920 - self.C.crop[0] * 2)
                    ud_bbox[1::2] = ud_bbox[1::2] / self.C.input_height * (1080 - self.C.crop[1] * 2)
                    ud_bbox[0::2] += self.C.crop[0]
                    ud_bbox[1::2] += self.C.crop[1]
                    int_bbox = ud_bbox.astype(np.int64)

                    img_ori = imageio.imread(file)
                    clip = img_ori[int_bbox[1] : int_bbox[3], int_bbox[0] : int_bbox[2]]
                    imageio.imwrite(file_out_dir / file.name, clip)

                    t.update()


def main():
    args = argparse.ArgumentParser()
    args.add_argument("config", type=str)
    args = args.parse_args(sys.argv[1:])

    with open(args.config, "r") as f:
        C = yaml.load(f, yaml.FullLoader)
        C = EasyDict(C)

    C.result_dir = Path(C.result_dir)
    for i in range(len(C.dataset.dirs)):
        C.dataset.dirs[i] = Path(C.dataset.dirs[i])

    tester = DetTester(C)
    for dpath in C.dataset.dirs:
        dl = tester.get_dataloader(dpath)
        tester.evaluate(dl, C.result_dir / "example" / f"{C.uid}_{dpath.name}")


if __name__ == "__main__":
    main()
