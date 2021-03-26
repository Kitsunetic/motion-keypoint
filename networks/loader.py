import torch
import torch.nn as nn

from .efficientdet import EfficientDet


class EfficientDetFinetune(EfficientDet):
    def __init__(self, model_name, pretrained=True, finetune=True):
        super().__init__(model_name, pretrained=pretrained)
        self.finetune = finetune

        if self.finetune:
            self.model.num_classes = 1
            self.model.classifier.num_classes = 1

            with torch.no_grad():
                conv1 = self.model.classifier.header.pointwise_conv.conv
                conv2 = nn.Conv2d(conv1.in_channels, conv1.out_channels // 90, 1)
                conv2.weight[:] = conv1.weight[: conv2.out_channels]
                conv2.bias[:] = conv1.bias[: conv2.out_channels]
                self.model.classifier.header.pointwise_conv.conv = conv2

                if self.model.classifier.header.norm:
                    bn1 = self.model.classifier.header.bn
                    bn2 = nn.BatchNorm2d(conv2.out_channels, momentum=0.01, eps=1e-3)
                    bn2.weight[:] = bn1.weight[: conv2.out_channels]
                    bn2.bias[:] = bn1.bias[: conv2.out_channels]
                    self.model.classifier.header.bn = bn2

    def unfreeze_tail(self):
        if not self.finetune:
            raise NotImplementedError()

        for p in self.parameters():
            p.requires_grad_(False)

        for p in self.model.classifier.header.parameters():
            p.requires_grad_(True)

    def unfreeze_head(self):
        if not self.finetune:
            raise NotImplementedError()

        for p in self.parameters():
            p.requires_grad_(True)

        for p in self.model.classifier.header.parameters():
            p.requires_grad_(False)

    def unfreeze(self):
        if not self.finetune:
            raise NotImplementedError()

        for p in self.parameters():
            p.requires_grad_(True)
