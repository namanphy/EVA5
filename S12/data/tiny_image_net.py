import os
import time
import imageio
import zipfile
import requests
import numpy as np
from io import BytesIO
from torch.utils.data import Dataset
from data.transformations import Transformations


class TinyImageNet(Dataset):
    def __init__(self, data_root=None, train=True, apply_transform=False, random_seed=123, train_test_split=0.7, padding=(0, 0), crop=(0, 0),
                 rotation=0, horizontal_flip=0.0, vertical_flip=0.0, cutout=0.0, cutout_hw_ratio=2):
        self.data_root = data_root

        if self.data_root is None:
            self.data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache')
        if not os.path.exists(self.data_root):
            os.makedirs(self.data_root)
        self.data_root = os.path.join(self.data_root, 'TinyImageNetData')
        self._download()
        self.train_test_split = train_test_split

        self._validate_inputs()

        self.train = train
        self.apply_transform = apply_transform

        self.data, self.targets = self._init_dataset()

        # Transformations
        self.mean = self._get_mean()
        self.std = self._get_std()
        self.padding = padding
        self.crop = crop
        self.rotation = rotation  # [int, int] or int
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.cutout = cutout
        self.cutout_hw_ratio = cutout_hw_ratio

        # Image size and transformation object
        self.image_size = np.transpose(self.data[0], (2, 0, 1)).shape
        self.transform = self._transform(self.train)

        self._image_indices = np.arange(len(self.targets))
        np.random.seed(random_seed)
        np.random.shuffle(self._image_indices)

        split_idx = int(len(self._image_indices) * self.train_test_split)
        self._image_indices = self._image_indices[:split_idx] if train else self._image_indices[split_idx:]

    def __len__(self):
        return len(self._image_indices)

    def __getitem__(self, idx):
        img_index = self._image_indices[idx]

        img_data = self.data[img_index]
        if self.apply_transform:
            img_data = self.transform(img_data)
        return img_data, self.targets[img_index]

    def _validate_inputs(self):
        if not (type(self.train_test_split) is float and 1 > self.train_test_split > 0):
            raise ValueError(f'train_test_split must be a float between 0 and 1, got {self.train_test_split} instead.')

    def _get_mean(self):
        return 0.5, 0.5, 0.5

    def _get_std(self):
        return 0.5, 0.5, 0.5

    def _get_id_dictionary(self):
        id_dict = {}
        for i, line in enumerate(open(os.path.join(self.data_root, 'wnids.txt'), 'r')):
            id_dict[line.replace('\n', '')] = i
        return id_dict

    def _init_dataset(self):
        print('starting loading data')
        id_dict = self._get_id_dictionary()
        data = []
        labels = []
        t = time.time()
        for key, value in id_dict.items():
            data += [imageio.imread(os.path.join(self.data_root, 'train', str(key), 'images', f'{key}_{i}.JPEG'),
                                    pilmode='RGB') for i in range(500)]
            labels_ = np.array([[0] * 200] * 500)
            labels_[:, value] = 1
            labels += labels_.tolist()

        for line in open(os.path.join(self.data_root, 'val', 'val_annotations.txt')):
            img_name, class_id = line.split('\t')[:2]
            data.append(imageio.imread(os.path.join(self.data_root, 'val', 'images', img_name), pilmode='RGB'))
            labels_ = np.array([[0] * 200])
            labels_[0, id_dict[class_id]] = 1
            labels += labels_.tolist()

        print('finished loading data, in {} seconds'.format(time.time() - t))
        return data, labels

    def _download(self):
        if not os.path.exists(self.data_root):
            print('Downloading dataset...')
            r = requests.get('http://cs231n.stanford.edu/tiny-imagenet-200.zip', stream=True)
            zf = zipfile.ZipFile(BytesIO(r.content))
            zf.extractall(os.path.dirname(self.data_root))
            zf.close()

            os.rename(
                os.path.join(os.path.dirname(self.data_root), 'tiny-imagenet-200'),
                self.data_root
            )
            print('Done.')
        else:
            print('Files already present.')

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
        id_dict = self._get_id_dictionary()
        all_classes = {}
        result = []
        for i, line in enumerate(open(os.path.join(self.data_root, 'words.txt'), 'r')):
            n_id, word = line.split('\t')[:2]
            all_classes[n_id] = word
        for key, value in id_dict.items():
            result.append(all_classes[key])
        return result
