import time
import copy
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets

DATA_DIR_PATH = "../Dataset/train"
MODEL = "densenet201"
EPOCH = 300
MODEL_WEIGHT_SAVE_PATH = f"../Models/{MODEL}_epoch{EPOCH}.pth"
TRAIN_RATIO = 0.8
BATCH_SIZE = 128
CLASSES = 2

class Train():
    def __init__(self, model, optimizer, criterion, dataloaders: dict, data_size: dict,
                 epoch: int, learning_rate: float):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataloaders = dataloaders
        self.data_size = data_size
        self.epoch = epoch
        self.learning_rate = learning_rate

    def do_train(self):
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

        # 途中経過保存用に、リストを持った辞書を作ります。
        loss_dict = {"train": [], "val": []}
        acc_dict = {"train": [], "val": []}

        best_acc = 0.0
        epoch_loss = 0
        epoch_acc = 0

        for epoch in tqdm(range(self.epoch)):
            if (epoch + 1) % 5 == 0:  # 5回に1回エポックを表示します。
                print('Epoch {}/{}'.format(epoch, self.epoch - 1))
                print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # 学習モード。dropoutなどを行う。
                else:
                    self.model.eval()  # 推論モード。dropoutなどを行わない。

                running_loss = 0.0
                running_corrects = 0

                for data in self.dataloaders[phase]:
                    inputs, labels = data

                    if torch.cuda.is_available():
                        inputs = inputs.cuda()
                        labels = labels.cuda()

                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss = self.criterion(outputs, labels)

                    if phase == 'train':
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels)

                    # リストに途中経過を格納
                    loss_dict[phase].append(epoch_loss)
                    acc_dict[phase].append(epoch_acc)

                epoch_loss = running_loss / self.data_size[phase]
                epoch_acc = running_corrects.item() / self.data_size[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
        print('Best val acc: {:.4f}'.format(best_acc))
        return best_model_wts