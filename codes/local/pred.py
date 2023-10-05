import glob
import os
import pandas as pd

from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets

from PIL import Image

class Pred():
    def __init__(self, model, data_transforms: dict, test_dir_path: str):
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model

        self.data_transforms = data_transforms
        self.test_dir_path = test_dir_path

    def do_prediction(self) -> pd.DataFrame:
        """学習して後にそのまま予測

        :return: pd.DataFrame
        """
        self.model.eval()
        file_names = []
        predictions = []

        for image_file_path in tqdm(glob.glob(self.test_dir_path)):
            file_name = os.path.basename(image_file_path)
            file_names.append(file_name)
            image = Image.open(image_file_path)
            image = self.data_transforms["val"](image)
            image = image.unsqueeze(0)
            if torch.cuda.is_available():
                image = image.cuda()

            with torch.no_grad():
                output = self.model(image)
                prediction = nn.functional.softmax(output, dim=1)
                prediction = prediction.data[0, 1].item()
                predictions.append(prediction)

        d = {'0': file_names, '1': predictions}
        dst_df = pd.DataFrame(data=d)

        return dst_df

    def do_prediction_by_best_weight(self, best_weight) -> pd.DataFrame:
        """最高精度の重みを読み込んで予測

        :param best_weight:
        :return:
        """
        self.model.load_state_dict(best_weight)
        self.model.eval()
        file_names = []
        predictions = []

        for image_file_path in tqdm(glob.glob(self.test_dir_path)):
            file_name = os.path.basename(image_file_path)
            file_names.append(file_name)
            image = Image.open(image_file_path)
            image = self.data_transforms["val"](image)
            image = image.unsqueeze(0)
            if torch.cuda.is_available():
                image = image.cuda()

            with torch.no_grad():
                output = self.model(image)
                prediction = nn.functional.softmax(output, dim=1)
                prediction = prediction.data[0, 1].item()
                predictions.append(prediction)

        d = {'0': file_names, '1': predictions}
        dst_df = pd.DataFrame(data=d)

        return dst_df