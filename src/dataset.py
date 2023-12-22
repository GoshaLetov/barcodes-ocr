import os
from typing import Union, Optional, Tuple

import numpy as np
import pandas as pd

import albumentations as albu
import cv2
from torch.utils.data import Dataset


TRANSFORM_TYPE = Union[albu.BasicTransform, albu.BaseCompose]


class BarCodeDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        data_folder: str,
        transforms: Optional[TRANSFORM_TYPE] = None,
    ):
        self.transforms = transforms

        self.crops = []
        self.codes = []
        for index, row in df.iterrows():
            image = cv2.imread(os.path.join(data_folder, row['filename']))[..., ::-1]
            x1 = int(row['x_from'])
            y1 = int(row['y_from'])
            x2 = int(row['x_from']) + int(row['width'])
            y2 = int(row['y_from']) + int(row['height'])
            crop = image[y1:y2, x1:x2]

            if crop.shape[0] > crop.shape[1]:
                crop = cv2.rotate(crop, 2)

            self.crops.append(crop)
            self.codes.append(str(row['code']))

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, str, int]:
        text = self.codes[idx]
        image = self.crops[idx]

        data = {
            'image': image,
            'text': text,
            'text_length': len(text),
        }

        if self.transforms:
            data = self.transforms(**data)

        return data['image'], data['text'], data['text_length']

    def __len__(self) -> int:
        return len(self.crops)
