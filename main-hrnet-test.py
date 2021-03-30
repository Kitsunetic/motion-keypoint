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
from tqdm import tqdm, trange

import datasets
import networks
import options
import utils
from datasets import get_pose_datasets
from losses import JointMSELoss, KeypointLoss, KeypointRMSE


class PoseTester:
    def __init__(self, config):
        self.C = config

        # Create Network
        if self.C.pose_model.name == "HRNet-W32":
            width = 32
        elif self.C.pose_model.name == "HRNet-W48":
            width = 48
        else:
            raise NotImplementedError()

        self.pose_model = networks.PoseHighResolutionNet(width)
        self.pose_model.load_state_dict(torch.load(f"networks/models/pose_hrnet_w{width}_384x288.pth"))

        final_layer = nn.Conv2d(width, 24, 1)
        with torch.no_grad():
            final_layer.weight[:17] = self.pose_model.final_layer.weight
            final_layer.bias[:17] = self.pose_model.final_layer.bias
            self.pose_model.final_layer = final_layer
        self.pose_model.cuda()

        if self.C.pose_model.pretrained is not None:
            self.pose_model.load_state_dict(torch.load(self.C.pose_model.pretrained)["model"])

        # Criterion
        self.criterion = KeypointLoss().cuda()
        self.criterion_rmse = KeypointRMSE().cuda()

        # Dataset
        files = sorted(list(self.C.dataset.dir.glob("*.jpg")))
        with open(self.C.dataset.dir / "info.json", "r") as f:
            info = json.load(f)

        self.ds_list = []
        for size in self.C.test.multiscale_test.input_sizes:
            for rotation in self.C.test.multiscale_test.rotations:
                for flip in self.C.test.multiscale_test.horizontal_flip:
                    ds = datasets.TestKeypointDataset(
                        files,
                        info,
                        normalize=self.C.dataset.normalize,
                        size=size,
                        rotation=rotation,
                        flip=flip,
                    )
                    self.ds_list.append(ds)

    @torch.no_grad()
    def multiscale_evaluate(self, file_out_dir):
        image_out_dir = Path(file_out_dir) / "image"
        heatmap_out_dir = Path(file_out_dir) / "heatmap"
        image_out_dir.mkdir(parents=True, exist_ok=True)
        heatmap_out_dir.mkdir(parents=True, exist_ok=True)
        self.pose_model.eval()

        keypoints = []
        with tqdm(total=len(self.ds_list[0]), ncols=100, file=sys.stdout) as t:
            for i in range(len(self.ds_list[0])):
                phmaps = []
                for _, ds in enumerate(self.ds_list):
                    file, img, offset, ratio, ori_size = ds[i]
                    file = Path(file)

                    # img_np = utils.denormalize(img).permute(1, 2, 0).mul(255).type(torch.uint8).numpy()
                    # imageio.imwrite("test.jpg", img_np)

                    phmap_ = self.pose_model(img.cuda(non_blocking=True).unsqueeze(0)).squeeze(0)
                    if ds.flip:
                        phmap_ = torch.flip(phmap_, (2,))
                    # ori_size = (int(phmap_.size(2) / ratio[0] * 4), int(phmap_.size(1) / ratio[1] * 4))
                    phmap_ = F.interpolate(phmap_.unsqueeze(0), ori_size[::-1], mode="bilinear", align_corners=True).squeeze(0)
                    if ds.rotation > 0:
                        phmap_ = torch.rot90(phmap_, 4 - ds.rotation, (1, 2))
                    phmaps.append(phmap_.cpu())

                phmap = phmap = torch.stack(phmaps)
                np.savez_compressed(heatmap_out_dir / f"{file.stem}.npz", phmap=phmap.numpy())
                if self.C.test.multiscale_test.voting_method == "mean":
                    pkeypoint = utils.heatmaps2keypoints(phmap.mean(0))
                elif self.C.test.multiscale_test.voting_method == "median":
                    pkeypoint = utils.heatmaps2keypoints(phmap.median(0).values)
                elif self.C.test.multiscale_test.voting_method == "key-mean":
                    pkeypoint = utils.heatmaps2keypoints(phmap).mean(0)
                elif self.C.test.multiscale_test.voting_method == "key-median":
                    pkeypoint = utils.heatmaps2keypoints(phmap).median(0).values
                else:
                    raise NotImplementedError(f"Unknown voting_method {self.C.test.multiscale_test.voting_method}")

                img = imageio.imread(file)
                imgk = utils.draw_keypoints(img, pkeypoint.type(torch.int))
                imageio.imwrite(image_out_dir / file.name, imgk)

                pkeypoint[:, 0] += offset[0]
                pkeypoint[:, 1] += offset[1]
                keypoints.append(pkeypoint)

                t.set_postfix_str(file.name)
                t.update()


def main():
    args = argparse.ArgumentParser()
    args.add_argument("config", type=str)
    args = args.parse_args(sys.argv[1:])

    with open(args.config, "r") as f:
        C = yaml.load(f, yaml.FullLoader)
        C = EasyDict(C)

    C.result_dir = Path(C.result_dir)
    C.result_dir.mkdir(parents=True, exist_ok=True)
    C.dataset.dir = Path(C.dataset.dir)

    trainer = PoseTester(C)
    trainer.multiscale_evaluate(C.result_dir / f"{C.uid}")


if __name__ == "__main__":
    main()
