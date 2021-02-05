import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class Transformations:
    """
    Albumentations transformations class.
    """

    def __init__(self, train=False, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), rotation=0, horizontal_flip=0.0,
                 cutout=0.0, cutout_height=None, cutout_width=None):
        self.train = train
        self.mean = mean
        self.std = std
        self.rotation = rotation  # [int, int] or int
        self.horizontal_flip = horizontal_flip
        self.cutout = cutout
        self.cutout_height = cutout_height
        self.cutout_width = cutout_width
        # self.transform = self.compose()

    # def compose(self):
    #     """
    #
    #     :return:
    #     """
        transforms_list = [
            ToTensorV2(),
            A.Normalize(self.mean, self.std, always_apply=True),
        ]

        augmentation_list = []
        if self.cutout > 0:
            augmentation_list.append(A.Cutout(num_holes=1, max_h_size=self.cutout_height, max_w_size=self.cutout_width,
                                              fill_value=[255.0*x for x in self.mean], always_apply=True, p=self.cutout))
        if self.horizontal_flip > 0:
            augmentation_list.append(A.HorizontalFlip(p=self.horizontal_flip))

        if self.rotation != 0:
            augmentation_list.append(A.Rotate(self.rotation))

        if self.train:
            transforms_list = transforms_list + augmentation_list

        print(transforms_list)
        self.transform = A.Compose(transforms_list)

    def __call__(self, image):
        image = np.array(image)
        image = self.transform(image=image)['image']
        return image
