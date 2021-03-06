from torchvision import datasets
from data.transformations import Transformations
import numpy as np


class CIFAR10:
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), padding=(0, 0), crop=(0, 0),
                 rotation=0, horizontal_flip=0.0, vertical_flip=0.0, cutout=0.0, cutout_hw_ratio=2):
        self.mean = mean
        self.std = std
        self.padding = padding
        self.crop = crop
        self.rotation = rotation  # [int, int] or int
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
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
            'padding': self.padding,
            'crop': self.crop,
            'rotation': self.rotation,
            'horizontal_flip': self.horizontal_flip,
            'vertical_flip': self.vertical_flip,
            'cutout': self.cutout,
            'cutout_height': self.image_size[1] // self.cutout_hw_ratio,
            'cutout_width': self.image_size[2] // self.cutout_hw_ratio
        }

        print("Transforms : ", args)
        return Transformations(**args)

    @property
    def classes(self):
        return ('airplane',	'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
                'horse', 'ship', 'truck')
