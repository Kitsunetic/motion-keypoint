import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class JointMSELoss(nn.Module):
    def __init__(self, use_target_weight=False):
        super(JointMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction="mean")
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight=None):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(heatmap_pred.mul(target_weight[:, idx]), heatmap_gt.mul(target_weight[:, idx]))
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class KeypointLoss(nn.Module):
    def __init__(self, joint=False, use_target_weight=False):
        super().__init__()
        self.criterion = nn.MSELoss(reduction="mean")
        self.joint = joint
        self.use_target_weight = use_target_weight

    def forward(self, x, y):
        x = x.flatten(2).flatten(0, 1)
        y = y.flatten(2).flatten(0, 1).argmax(1)
        loss1 = F.cross_entropy(x, y)

        if self.joint:
            loss2 = self.joint_mse_loss(x, y)
            return loss1 + loss2
        else:
            return loss1

    def joint_mse_loss(self, output, target, target_weight=None):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(heatmap_pred.mul(target_weight[:, idx]), heatmap_gt.mul(target_weight[:, idx]))
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class KeypointRMSE(nn.Module):
    @torch.no_grad()
    def forward(self, pred_heatmaps: torch.Tensor, real_heatmaps: torch.Tensor, ratios: torch.Tensor):
        W = pred_heatmaps.size(3)
        pred_positions = pred_heatmaps.flatten(2).argmax(2)
        real_positions = real_heatmaps.flatten(2).argmax(2)
        pred_positions = torch.stack((pred_positions // W, pred_positions % W), 2).type(torch.float32)
        real_positions = torch.stack((real_positions // W, real_positions % W), 2).type(torch.float32)
        # print(pred_positions.shape, real_positions.shape, ratios.shape)
        pred_positions *= 4 / ratios.unsqueeze(1)  # position: (B, 24, 2), ratio: (B, 2)
        real_positions *= 4 / ratios.unsqueeze(1)
        loss = (pred_positions - real_positions).square().mean().sqrt()

        return loss


class KeypointBCELoss(nn.BCELoss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(input.sigmoid(), target)


class SigmoidMAE(nn.L1Loss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(input.sigmoid(), target)


class SigmoidKLDivLoss(nn.KLDivLoss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(input.sigmoid(), target)


class AWing(nn.Module):
    # https://github.com/SeungyounShin/Adaptive-Wing-Loss-for-Robust-Face-Alignment-via-Heatmap-Regression/blob/1f4d6528ebf55fb65f2f82104c3f86d761010b3e/losses/loss.py#L5

    def __init__(self, alpha=2.1, omega=14, epsilon=1, theta=0.5):
        super().__init__()
        self.alpha = float(alpha)
        self.omega = float(omega)
        self.epsilon = float(epsilon)
        self.theta = float(theta)

    def forward(self, y_pred, y):
        lossMat = torch.zeros_like(y_pred)
        A = (
            self.omega
            * (1 / (1 + (self.theta / self.epsilon) ** (self.alpha - y)))
            * (self.alpha - y)
            * ((self.theta / self.epsilon) ** (self.alpha - y - 1))
            / self.epsilon
        )
        C = self.theta * A - self.omega * torch.log(1 + (self.theta / self.epsilon) ** (self.alpha - y))
        case1_ind = torch.abs(y - y_pred) < self.theta
        case2_ind = torch.abs(y - y_pred) >= self.theta
        lossMat[case1_ind] = self.omega * torch.log(
            1 + torch.abs((y[case1_ind] - y_pred[case1_ind]) / self.epsilon) ** (self.alpha - y[case1_ind])
        )
        lossMat[case2_ind] = A[case2_ind] * torch.abs(y[case2_ind] - y_pred[case2_ind]) - C[case2_ind]
        return lossMat.mean()
