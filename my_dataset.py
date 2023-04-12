import os
from configparser import ConfigParser
import cv2
import numpy as np
import torch
from albumentations import Compose, Normalize, Resize
from torch.utils.data import Dataset
from utils.data_utils import run_length_decode

config_parser = ConfigParser()
config_parser.read('/home/chenk/model_train/kaggle/SIIM_ACR_Pneumothorax_Segmentation/mysql.cfg')
cfg = config_parser['default']


class My_Dataset(torch.utils.data.Dataset):
    def __init__(self, df, data_path, label=True, transform=None):
        self.df = df
        self.root = data_path
        self.label = label
        self.transform = transform
        self.images_path = [os.path.join(self.root, img_id + '.png') for img_id in self.df['ImageId']]
        self.images_rle_mask = df[' EncodedPixels'].tolist()

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        image_path = self.images_path[item]
        image = cv2.imread(image_path)

        if self.label:
            mask = np.zeros([1024, 1024])
            rle_mask = self.images_rle_mask[item]
            if rle_mask != '-1':
                mask += run_length_decode(rle_mask)
            mask = (mask >= 1).astype('float32')
            if self.transform:
                data = self.transform(image=image, mask=mask)
                image = data['image']
                image = np.transpose(image, (2, 0, 1))
                mask = data['mask']
            mask = np.expand_dims(mask, axis=0)
            return torch.tensor(image, dtype=torch.float), torch.tensor(mask, dtype=torch.float)
        else:
            if self.transform:
                data = self.transform(image=image)
                image = data['image']
                image = np.transpose(image, (2, 0, 1))
            return torch.tensor(image, dtype=torch.float)


class TestDataset(Dataset):
    def __init__(self, root, df, size, mean, std):
        self.root = root
        self.size = size
        self.fnames = list(df["ImageId"])
        self.num_samples = len(self.fnames)
        self.transform = Compose(
            [
                Normalize(mean=mean, std=std, p=1),
                Resize(size, size)
            ]
        )

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        path = os.path.join(self.root, fname + '.png')
        image = cv2.imread(path)
        images = self.transform(image=image)["image"]
        images = np.transpose(images, (2, 0, 1))
        return torch.tensor(images, dtype=torch.float)

    def __len__(self):
        return self.num_samples
