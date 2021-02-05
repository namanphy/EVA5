from torchvision import datasets
from data.transformations import Transformations
import torch
import numpy as np


class CIFAR10:
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), rotation=0, horizontal_flip=0.0,
                 cutout=0.0, cutout_hw_ratio=2):
        self.mean = mean
        self.std = std
        self.rotation = rotation  # [int, int] or int
        self.horizontal_flip = horizontal_flip
        self.cutout = cutout
        self.cutout_hw_ratio = cutout_hw_ratio

        self.transform = None
        self.sample_data = self.download()
        self.image_size = np.transpose(self.sample_data.data[0], (2, 0, 1)).shape

    def download(self, loc= '../data', train=True, apply_transform=False):
        if apply_transform:
            self.transform = self._transform(train=train)
        return datasets.CIFAR10(loc, train=train, download=True, transform=self.transform)

    def _transform(self, train=True):
        args = {
            'train': train,
            'mean': self.mean,
            'std': self.std,
            'rotation': self.rotation,
            'horizontal_flip': self.horizontal_flip,
            'cutout': self.cutout,
            'cutout_height': self.image_size[0] // 2,
            'cutout_width': self.image_size[1] // 2
        }

        print("Transforms : ", args)
        transformation_obj = Transformations(**args)
        return transformation_obj.compose()
