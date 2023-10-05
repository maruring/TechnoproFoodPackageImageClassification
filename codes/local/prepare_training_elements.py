import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim

from typing import Union

class PrepareTrainingElements():
    """学習に必要なモデル、ファインチューニング、オプティマイザー、クライテリオンなどを取得する
    """
    def __init__(self, model_name: str, class_num: int):
        self.model_name = model_name
        self.class_num = class_num

    def get_fine_tuned_model(self):
        # modelの調整
        if self.model_name == "ResNet":
            model = models.resnext50_32x4d(pretrained=True)
            for p in model.parameters():
                p.requires_grad = False
            classifier = nn.Sequential(
                nn.Linear(in_features=2048, out_features=1024),
                nn.ReLU(),
                nn.Dropout(p=0.4),
                nn.Linear(in_features=1024, out_features=self.class_num),
                nn.LogSoftmax(dim=1)
            )
            model.fc = classifier

        elif self.model_name == "DenseNet":
            model = models.densenet201(pretrained=True)
            for p in model.parameters():
                p.requires_grad = False
            classifier = nn.Sequential(
                nn.Linear(in_features=1920, out_features=1024),
                nn.ReLU(),
                nn.Dropout(p=0.4),
                nn.Linear(in_features=1024, out_features=self.class_num),
                nn.LogSoftmax(dim=1)
            )
            model.classifier = classifier

        elif self.model_name == "RegNet":
            model = models.regnet_x_8gf(pretrained=True)
            for p in model.parameters():
                p.requires_grad = False
            classifier = nn.Sequential(
                nn.Linear(in_features=1920, out_features=1024),
                nn.ReLU(),
                nn.Dropout(p=0.4),
                nn.Linear(in_features=1024, out_features=self.class_num),
                nn.LogSoftmax(dim=1)
            )
            model.fc = classifier

        else:
            raise KeyError

        return model

    def get_optimizer(self, model_parameters, lr, weight_decay, momentum):
        if self.model_name == "ResNet":
            optimizer = torch.optim.SGD(model_parameters, lr=lr, weight_decay=weight_decay, momentum=momentum)
        else:
            optimizer = optim.Adam(model_parameters, lr=lr, weight_decay=weight_decay)

        return optimizer

    def get_criterion(self):
        return nn.CrossEntropyLoss()