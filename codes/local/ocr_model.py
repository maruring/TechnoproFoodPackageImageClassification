import copy

import numpy as np
import pandas as pd
import cv2 as cv
from tqdm import tqdm
import easyocr


import torch
import torch.nn as nn

class ReadPackageText():
    """パッケージに記載されている文字を読み取るクラス
    読取精度向上のための画像処理も担当する

    """
    def __init__(self):
        self.reader = easyocr.Reader(["ja", "en"], gpu=True)
        self.font = cv.FONT_HERSHEY_SIMPLEX

    def improve_image_process(self, src_image) -> np.array:
        """OCR精度を向上させるための画像処理関係

        :param src_image: 3ch
        :return: np.array
        """
        temp_image = copy.deepcopy(src_image)
        dst_image = temp_image
        return dst_image

    def read_text(self, src_image: np.array) -> list[str]:
        """文字を読取

        :param src_image:
        :return: list[str]
        """
        dst_texts = []
        results = self.reader.readtext(src_image)
        for result in results:
            read_text = result[1]
            dst_texts.append(read_text)

        return dst_texts


if __name__ == "__main__":
    train_info_path = "../Dataset/train.csv"
    train_info = pd.read_csv(train_info_path)
    package_text_reader = ReadPackageText()
    image_names = []
    labels = []
    texts = []

    for index, row in train_info.iterrows():
        print(f"index is {index}")
        image_name = row["image_name"]
        label = row["label"]

        image_path = f"../Dataset/train/{label}/{image_name}"
        image = cv.imread(image_path)
        letters: list[str] = package_text_reader.read_text(src_image=image)
        if len(letters) == 0:
            letters.append("Dead")
        else:
            pass
        text = ''.join(letters)

        image_names.append(image_name)
        labels.append(label)
        texts.append(text)

    train_data = pd.DataFrame()
    train_data["image_name"] = image_names
    train_data["text"] = texts
    train_data["label"] = labels
    train_data.to_csv("../Dataset/NLP_Model_train.csv", index=False, header=True, encoding="shift-jis")
