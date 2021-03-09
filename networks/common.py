import torch
import torch.nn as nn
from torch import nn
from torchvision.models import mobilenet_v2
from torchvision.models.detection import KeypointRCNN, keypointrcnn_resnet50_fpn
from torchvision.ops import MultiScaleRoIAlign
from tqdm import tqdm
import requests

import networks

MODEL_NAMES = ["우주대마왕", "keypointrcnn_resnet50_fpn_finetune", "HRNet32_finetune"]


def _requires_grad_(model: nn.Module, mode: bool):
    for p in model.parameters():
        p.requires_grad_(mode)


def get_model(model_name) -> nn.Module:
    if model_name == "우주대마왕":
        backbone = mobilenet_v2(pretrained=True).features
        backbone.out_channels = 1280
        roi_pooler = MultiScaleRoIAlign(featmap_names=["0"], output_size=7, sampling_ratio=2)

        keypoint_roi_pooler = MultiScaleRoIAlign(featmap_names=["0"], output_size=14, sampling_ratio=2)

        model = KeypointRCNN(
            backbone,
            num_classes=2,
            num_keypoints=24,
            box_roi_pool=roi_pooler,
            keypoint_roi_pool=keypoint_roi_pooler,
        )
    elif model_name == "keypointrcnn_resnet50_fpn_finetune":
        model = keypointrcnn_resnet50_fpn(pretrained=True, progress=False)
        _requires_grad_(model, False)

        m = nn.ConvTranspose2d(512, 24, 4, 2, 1)
        with torch.no_grad():
            m.weight[:, :17] = model.roi_heads.keypoint_predictor.kps_score_lowres.weight
            m.bias[:17] = model.roi_heads.keypoint_predictor.kps_score_lowres.bias
        model.roi_heads.keypoint_predictor.kps_score_lowres = m
    elif model_name == "HRNet32_finetune_single":
        model = networks.PoseHighResolutionNet(width=32, num_keypoints=17)
        # TODO 파일 다운받아서 할 수 있도록 하기. 현재는 구글드라이브라서 주소로 못받음...
        ckpt = torch.load("networks/models/pose_hrnet_w32_384x288.pth")
        model.load_state_dict(ckpt)
        _requires_grad_(model, False)

        layer = nn.Conv2d(model.final_layer.in_channels, 24, 1)
        with torch.no_grad():
            layer.weight[:17] = model.final_layer.weight
            layer.bias[:17] = model.final_layer.bias
        model.final_layer = layer
    else:
        raise NotImplementedError(f"`model_name` must be one of {MODEL_NAMES}")

    return model.cuda()
