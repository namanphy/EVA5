import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class Transformations:
    """
    Albumentations transformations class.
    """

    def __init__(self, train=False, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), padding=(0, 0), crop=(0, 0),
                 rotation=0, horizontal_flip=0.0, vertical_flip=0.0, cutout=0.0, cutout_height=None, cutout_width=None):
        self.train = train
        self.mean = mean
        self.std = std
        self.padding = padding
        self.crop = crop
        self.rotation = rotation  # [int, int] or int
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.cutout = cutout
        self.cutout_height = cutout_height
        self.cutout_width = cutout_width
        # self.transform = self.compose()

        transforms_list = [
            A.Normalize(self.mean, self.std, always_apply=True),
            ToTensorV2()
        ]

        augmentation_list = []
        if sum(self.padding) > 0:
            augmentation_list.append(A.PadIfNeeded(padding[0], padding[1], always_apply=True))
        if sum(self.crop) > 0:
            augmentation_list.append(A.RandomCrop(self.crop[0], self.crop[1]))

        if self.horizontal_flip > 0:
            augmentation_list.append(A.HorizontalFlip(p=self.horizontal_flip))

        if self.vertical_flip > 0:
            augmentation_list.append(A.VerticalFlip(p=self.vertical_flip))

        if self.rotation != 0:
            augmentation_list.append(A.Rotate(self.rotation))

        if self.cutout > 0:
            augmentation_list.append(A.Cutout(num_holes=1, max_h_size=self.cutout_height, max_w_size=self.cutout_width,
                                              fill_value=[255.0*x for x in self.mean], always_apply=True, p=self.cutout))

        if self.train:
            transforms_list = augmentation_list + transforms_list

        self.transform = A.Compose(transforms_list)

    def __call__(self, image):
        image = np.array(image)
        image = self.transform(image=image)['image']
        return image
