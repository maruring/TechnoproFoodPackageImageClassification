import os
import datetime

from sklearn.model_selection import KFold

import torch
from torchvision import models, transforms, datasets

from train import Train
from pred import Pred
from prepare_training_elements import PrepareTrainingElements
from get_original_model import MyEnsembleModel

DATA_DIR_PATH = "../Dataset/train"
TEST_DIR_PATH = "../Dataset/test/*"
MODEL = "DenseNet"
EPOCH = 120
RANDOM_STATE = 2023
t_delta = datetime.timedelta(hours=9)
JST = datetime.timezone(t_delta, 'JST')
dt_now = datetime.datetime.now(JST).strftime('%Y%m%d')
if not os.path.isdir(f"../output/{dt_now}"):
    os.mkdir(f"../output/{dt_now}")
TRAIN_RATIO = 0.8
BATCH_SIZE = 128
CLASSES = 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.01

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

if __name__ == "__main__":
    folds = KFold(shuffle=True, n_splits=5, random_state=RANDOM_STATE)
    data = datasets.ImageFolder(root=DATA_DIR_PATH, transform=data_transforms["train"])
    train_size = int(TRAIN_RATIO * len(data))
    val_size = len(data) - train_size
    data_size = {"train": train_size, "val": val_size}
    # data_train, data_val = torch.utils.data.random_split(data, [train_size, val_size],)
    for i_fold, (train_idx, val_idx) in enumerate(folds.split(data)): # クロスバリデーション
        data_train = torch.utils.data.Subset(data, train_idx)
        data_val = torch.utils.data.Subset(data, val_idx)

        train_loader = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = torch.utils.data.DataLoader(data_val, batch_size=BATCH_SIZE, shuffle=False)
        dataloaders = {"train": train_loader, "val": val_loader}

        MODEL_WEIGHT_SAVE_PATH = f"../Models/{dt_now}_{MODEL}_epoch{EPOCH}_{i_fold}.pth"
        PREDICTION_FILE_SAVE_PATH = f"../output/{dt_now}/{MODEL}_epoch{EPOCH}_{i_fold}.csv"
        models_prepeaer = PrepareTrainingElements(model_name=MODEL, class_num=CLASSES)

    model_0 = models_prepeaer.get_fine_tuned_model()
    model_0.load_state_dict(torch.load("../Models/20230904_DenseNet_epoch120_0.pth"))
    model_1 = models_prepeaer.get_fine_tuned_model()
    model_1.load_state_dict(torch.load("../Models/20230904_DenseNet_epoch120_1.pth"))
    model_2 = models_prepeaer.get_fine_tuned_model()
    model_2.load_state_dict(torch.load("../Models/20230904_DenseNet_epoch120_2.pth"))
    model_3 = models_prepeaer.get_fine_tuned_model()
    model_3.load_state_dict(torch.load("../Models/20230904_DenseNet_epoch120_3.pth"))
    model_4 = models_prepeaer.get_fine_tuned_model()
    model_4.load_state_dict(torch.load("../Models/20230904_DenseNet_epoch120_4.pth"))
    model = MyEnsembleModel(modelA=model_0, modelB=model_1, modelC=model_2, modelD=model_3, modelE=model_4, input=2)


    """
    optimizer = models_prepeaer.get_optimizer(model_parameters=model.parameters(),
                                              lr=LEARNING_RATE,
                                              weight_decay=WEIGHT_DECAY,
                                              momentum=MOMENTUM)
    criterion = models_prepeaer.get_criterion()

    trainer = Train(model=model, optimizer=optimizer, criterion=criterion, dataloaders=dataloaders,
                    data_size=data_size, epoch=EPOCH, learning_rate=LEARNING_RATE)
    best_model_wts = trainer.do_train()
    del trainer, criterion, optimizer, models_prepeaer
    torch.save(best_model_wts, MODEL_WEIGHT_SAVE_PATH)
    """

    predictor = Pred(model=model, data_transforms=data_transforms, test_dir_path=TEST_DIR_PATH)
    prediction_df = predictor.do_prediction()
    prediction_df.to_csv(PREDICTION_FILE_SAVE_PATH, index=False, header=False)
    del predictor, model