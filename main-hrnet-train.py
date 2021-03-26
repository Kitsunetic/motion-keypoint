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
from PIL import Image
from sklearn.model_selection import KFold
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

import networks
import options
import utils
from datasets import get_pose_datasets
from losses import JointMSELoss, KeypointLoss, KeypointRMSE


class PoseTrainer:
    def __init__(self, config, fold, checkpoint=None):
        self.C = config
        self.fold = fold

        # Create Network
        if self.C.pose_model == "HRNet-W32":
            width = 32
        elif self.C.pose_model == "HRNet-W48":
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

        # Criterion
        self.criterion = KeypointLoss().cuda()
        self.criterion_rmse = KeypointRMSE().cuda()

        # Optimizer
        if self.C.SAM:
            self.optimizer = utils.SAM(self.pose_model.parameters(), optim.AdamW, lr=self.C.lr)
        else:
            self.optimizer = optim.AdamW(self.pose_model.parameters(), lr=self.C.lr)

        # Scheduler
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=3, verbose=True)

        self.epoch = 1
        self.best_loss = math.inf
        self.earlystop_cnt = 0

        # Dataset
        self.dl_train, self.dl_valid, self.dl_test = get_pose_datasets(self.C, self.fold)

        # Load Checkpoint
        if checkpoint is not None:
            self.load(checkpoint)

    def save(self, path):
        torch.save(
            {
                "model": self.pose_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": self.epoch,
                "best_loss": self.best_loss,
                "earlystop_cnt": self.earlystop_cnt,
            },
            path,
        )

    def load(self, path):
        print("Load pretrained", path)
        ckpt = torch.load(path)
        self.pose_model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epoch = ckpt["epoch"] + 1
        self.best_loss = ckpt["best_loss"]
        self.earlystop_cnt = ckpt["earlystop_cnt"]

    def close(self):
        self.logger.close()

    def train_loop(self):
        self.pose_model.train()

        meanloss, meanrmse = utils.AverageMeter(), utils.AverageMeter()
        with tqdm(
            total=len(self.dl_train.dataset),
            ncols=100,
            leave=False,
            file=sys.stdout,
            desc=f"Train {self.epoch:03d}",
        ) as t:
            for files, imgs, keypoints, target_heatmaps, ratios in self.dl_train:
                imgs_, target_heatmaps_ = imgs.cuda(non_blocking=True), target_heatmaps.cuda(non_blocking=True)
                pred_heatmaps_ = self.pose_model(imgs_)
                loss = self.criterion(pred_heatmaps_, target_heatmaps_)
                rmse = self.criterion_rmse(pred_heatmaps_, target_heatmaps_, ratios.cuda(non_blocking=True))

                self.optimizer.zero_grad()
                loss.backward()
                if isinstance(self.optimizer, utils.SAM):
                    self.optimizer.first_step()
                    self.criterion(self.pose_model(imgs_), target_heatmaps_).backward()
                    self.optimizer.second_step()
                else:
                    self.optimizer.step()

                meanloss.update(loss.item(), len(files))
                meanrmse.update(rmse.item(), len(files))
                t.set_postfix_str(f"loss: {loss.item():.6f}, rmse: {rmse.item():.6f}", refresh=False)
                t.update(len(imgs))

        return meanloss(), meanrmse()

    @torch.no_grad()
    def valid_loop(self):
        self.pose_model.eval()

        meanloss, meanrmse = utils.AverageMeter(), utils.AverageMeter()
        with tqdm(
            total=len(self.dl_valid.dataset),
            ncols=100,
            leave=False,
            file=sys.stdout,
            desc=f"Valid {self.epoch:03d}",
        ) as t:
            for files, imgs, keypoints, target_heatmaps, ratios in self.dl_valid:
                imgs_, target_heatmaps_ = imgs.cuda(non_blocking=True), target_heatmaps.cuda(non_blocking=True)
                pred_heatmaps_ = self.pose_model(imgs_)
                loss = self.criterion(pred_heatmaps_, target_heatmaps_)
                rmse = self.criterion_rmse(pred_heatmaps_, target_heatmaps_, ratios.cuda(non_blocking=True))

                meanloss.update(loss.item(), len(files))
                meanrmse.update(rmse.item(), len(files))
                t.set_postfix_str(f"loss: {loss.item():.6f}, rmse: {rmse.item():.6f}", refresh=False)
                t.update(len(imgs))

        return meanloss(), meanrmse()

    def finetune_step1(self):
        self.C.log.info("Finetune Step 1")
        self.pose_model.freeze_head()
        for self.epoch in range(self.epoch, self.C.step1_epoch + 1):
            tloss, trmse = self.train_loop()
            vloss, vrmse = self.valid_loop()

            self.C.log.info(
                f"Epoch: {self.epoch:03d},",
                f"loss: {tloss:.6f};{vloss:.6f},",
                f"rmse {trmse:.6f};{vrmse:.6f}",
            )
            self.C.log.flush()

            if self.best_loss > vloss:
                self.best_loss = vloss
                self.save(self.C.result_dir / f"ckpt-{self.C.uid}_{self.fold}.pth")

    def finetune_step2(self):
        self.C.log.info("Finetune Step 2")
        self.pose_model.freeze_tail()
        for self.epoch in range(self.epoch, self.C.step2_epoch + 1):
            tloss, trmse = self.train_loop()
            vloss, vrmse = self.valid_loop()

            self.C.log.info(
                f"Epoch: {self.epoch:03d},",
                f"loss: {tloss:.6f};{vloss:.6f},",
                f"rmse {trmse:.6f};{vrmse:.6f}",
            )
            self.C.log.flush()

            if self.best_loss > vloss:
                self.best_loss = vloss
                self.save(self.C.result_dir / f"ckpt-{self.C.uid}_{self.fold}.pth")

    def finetune_step3(self):
        self.C.log.info("Finetune Step 3")
        self.pose_model.unfreeze_all()
        for self.epoch in range(self.epoch, self.C.step3_epoch + 1):
            tloss, trmse = self.train_loop()
            vloss, vrmse = self.valid_loop()

            self.C.log.info(
                f"Epoch: {self.epoch:03d},",
                f"loss: {tloss:.6f};{vloss:.6f},",
                f"rmse {trmse:.6f};{vrmse:.6f}",
            )
            self.C.log.flush()
            self.scheduler.step(vloss)

            if self.best_loss > vloss:
                self.best_loss = vloss
                self.earlystop_cnt = 0
                self.save(self.C.result_dir / f"ckpt-{self.C.uid}_{self.fold}.pth")
            elif self.earlystop_cnt >= 10:
                self.C.log.info(f"Stop training at epoch", self.epoch)
                break
            else:
                self.earlystop_cnt += 1

    def fit(self):
        self.finetune_step1()
        self.epoch += 1
        # torch.cuda.empty_cache()
        self.load(self.C.result_dir / f"ckpt-{self.C.uid}_{self.fold}.pth")

        self.finetune_step2()
        self.epoch += 1
        # torch.cuda.empty_cache()
        self.load(self.C.result_dir / f"ckpt-{self.C.uid}_{self.fold}.pth")

        self.finetune_step3()
        self.epoch += 1
        # torch.cuda.empty_cache()
        self.load(self.C.result_dir / f"ckpt-{self.C.uid}_{self.fold}.pth")

    @torch.no_grad()
    def evaluate(self, dl: DataLoader, file_out_dir=None):
        self.pose_model.eval()

        meanloss, meanrmse = utils.AverageMeter(), utils.AverageMeter()
        with tqdm(total=len(dl.dataset), ncols=100, file=sys.stdout) as t:
            for files, imgs, keypoints, target_heatmaps, ratios in dl:
                imgs_, target_heatmaps_ = imgs.cuda(non_blocking=True), target_heatmaps.cuda(non_blocking=True)
                pred_heatmaps_ = self.pose_model(imgs_)
                loss = self.criterion(pred_heatmaps_, target_heatmaps_)
                rmse = self.criterion_rmse(pred_heatmaps_, target_heatmaps_, ratios.cuda(non_blocking=True))

                if file_out_dir is not None:
                    file_out_dir = Path(file_out_dir)
                    file_out_dir.mkdir(parents=True, exist_ok=True)

                    for file, img, pred_heatmap, target_heatmap, ratio in zip(
                        files, imgs, pred_heatmaps_.cpu(), target_heatmaps, ratios
                    ):
                        file = Path(file)

                        pred_keypoint = utils.heatmaps2keypoints(pred_heatmap).type(torch.float32)
                        target_keypoint = utils.heatmaps2keypoints(target_heatmap).type(torch.float32)
                        pred_keypoint = pred_keypoint * 4 / ratio.view(1, 2)
                        target_keypoint = target_keypoint * 4 / ratio.view(1, 2)
                        keypoint_rmse = (pred_keypoint - target_keypoint).square().mean().sqrt()

                        img_np = utils.denormalize(img).permute(1, 2, 0).mul(255).type(torch.uint8).numpy()
                        img_np = np.array(Image.fromarray(img_np))
                        img_np = cv2.resize(img_np, (int(img_np.shape[1] / ratio[0]), int(img_np.shape[0] / ratio[1])))
                        img_pred_keypoint = utils.draw_keypoints(img_np, pred_keypoint.type(torch.int64))
                        img_target_keypoint = utils.draw_keypoints(img_np, target_keypoint.type(torch.int64))

                        imageio.imwrite(file_out_dir / f"{file.stem}_{keypoint_rmse:.2f}.jpg", img_pred_keypoint)
                        imageio.imwrite(file_out_dir / file.name, img_target_keypoint)

                    meanloss.update(loss.item())
                    meanrmse.update(rmse.item())
                    t.set_postfix_str(f"loss: {loss.item():.6f}, rmse: {rmse.item():.6f}", refresh=False)
                    t.update(len(imgs))

        return meanloss(), meanrmse()


def main():
    args = argparse.ArgumentParser()
    args.add_argument("config_file", type=str)
    args = args.parse_args(sys.argv[1:])

    config = options.load_config(args.config_file)
    for fold, checkpoint in zip(config.folds, config.checkpoints):
        config.log.file.write("===============================================================")
        config.log.info("Fold", fold)
        trainer = PoseTrainer(config, fold, checkpoint)
        trainer.fit()

        # validation exmaple 이미지 저장
        trainer.evaluate(trainer.dl_valid, config.result_dir / "example" / f"valid_{fold}")


if __name__ == "__main__":
    main()
