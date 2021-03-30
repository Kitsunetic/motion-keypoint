import argparse
import json
import math
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
from sklearn.model_selection import KFold
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

import networks
import options
import utils
from datasets import get_pose_datasets
from losses import JointMSELoss, KeypointLoss, KeypointRMSE


class PoseTester:
    def __init__(self, config, fold, checkpoint=None):
        self.C = config
        self.fold = fold

        # Create Network
        if self.C.pose_model.name == "HRNet-W32":
            width = 32
        elif self.C.pose_model.name == "HRNet-W48":
            width = 48
        else:
            raise NotImplementedError()

        self.pose_model = networks.PoseHighResolutionNet(width)
        if checkpoint is None:
            self.pose_model.load_state_dict(torch.load(f"networks/models/pose_hrnet_w{width}_384x288.pth"))
        else:
            self.pose_model.load_state_dict(torch.load(checkpoint)["model"])

        final_layer = nn.Conv2d(width, 24, 1)
        with torch.no_grad():
            final_layer.weight[:17] = self.pose_model.final_layer.weight
            final_layer.bias[:17] = self.pose_model.final_layer.bias
            self.pose_model.final_layer = final_layer
        self.pose_model.cuda()

        # Criterion
        self.criterion = KeypointLoss().cuda()
        self.criterion_rmse = KeypointRMSE().cuda()

        # Dataset
        self.ds_list = []
        for dpath in self.C.dataset.dirs:
            datasets.TestKeypointDataset()

    @torch.no_grad()
    def valid_loop(self, file_out_dir):
        file_out_dir = Path(file_out_dir)
        file_out_dir.mkdir(parents=True, exist_ok=True)
        self.pose_model.eval()

        meanloss, meanrmse = utils.AverageMeter(), utils.AverageMeter()
        with tqdm(total=len(self.dl_valid.dataset), ncols=100, file=sys.stdout, desc=f"ValidLoop") as t:
            for files, imgs, keypoints, thmaps, ratios in self.dl_valid:
                imgs_, thmaps_ = imgs.cuda(non_blocking=True), thmaps.cuda(non_blocking=True)
                phmaps_ = self.pose_model(imgs_)

                loss = self.criterion(phmaps_, thmaps_)
                rmse = self.criterion_rmse(phmaps_, thmaps_, ratios.cuda(non_blocking=True))
                meanloss.update(loss.item(), len(files))
                meanrmse.update(rmse.item(), len(files))

                if file_out_dir is not None:
                    for file, phmap, thmap, ratio in zip(files, phmaps_.cpu(), thmaps, ratios):
                        file = Path(file)
                        img_np = imageio.imread(file)
                        pkey = utils.heatmaps2keypoints(phmap).type(torch.float)
                        tkey = utils.heatmaps2keypoints(thmap).type(torch.float)
                        pkey /= ratio.view(1, 2)
                        tkey /= ratio.view(1, 2)
                        pimg = utils.draw_keypoints(img_np, pkey)
                        timg = utils.draw_keypoints(img_np, tkey)
                        imageio.imwrite(file_out_dir / f"{file.stem}p-{rmse:.2f}.jpg", pimg)
                        imageio.imwrite(file_out_dir / f"{file.stem}t-{rmse:.2f}.jpg", timg)
                        t.update()
                else:
                    t.update(len(imgs))

        return meanloss(), meanrmse()

    @torch.no_grad()
    def test_loop(self, file_out_dir):
        file_out_dir = Path(file_out_dir)
        file_out_dir.mkdir(parents=True, exist_ok=True)
        self.pose_model.eval()

        with tqdm(total=len(self.dl_test.dataset), ncols=100, file=sys.stdout, desc=f"TestLoop") as t:
            for files, imgs, ratios in self.dl_test:
                imgs_ = imgs.cuda(non_blocking=True)
                phmaps_ = self.pose_model(imgs_)

                if file_out_dir is not None:
                    for file, phmap, ratio in zip(files, phmaps_.cpu(), ratios):
                        file = Path(file)
                        t.set_postfix_str(file.stem)
                        img_np = imageio.imread(file)
                        pkey = utils.heatmaps2keypoints(phmap).type(torch.float)
                        pkey /= ratio.view(1, 2)
                        pimg = utils.draw_keypoints(img_np, pkey)
                        imageio.imwrite(file_out_dir / f"{file.stem}.jpg", pimg)
                        t.update()
                else:
                    t.update(len(imgs))


# TODO: multi-scale test + flip test + rotate test


def main():
    args = argparse.ArgumentParser()
    args.add_argument("config", type=str)
    args = args.parse_args(sys.argv[1:])

    with open(args.config, "r") as f:
        C = yaml.load(f, yaml.FullLoader)
        C = EasyDict(C)

    C.result_dir = Path(C.result_dir)
    C.result_dir.mkdir(parents=True, exist_ok=True)
    for i in range(len(C.dataset.dirs)):
        C.dataset.dirs[i] = Path(C.dataset.dirs[i])

    trainer = PoseTester(C, 1, C.pose_model.pretrained)
    trainer.valid_loop(C.result_dir / f"{C.uid}_valid_1")
    trainer.test_loop(C.result_dir / f"{C.uid}_test_1")


if __name__ == "__main__":
    main()
