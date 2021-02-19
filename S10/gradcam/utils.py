import cv2
import matplotlib.cm as cm
import numpy as np
from torchvision import transforms


def save_gradcam(filename, gcam, raw_image, save_as_file=False):
    print(f"\t Generating Image : {filename}")
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2

    if save_as_file:
        cv2.imwrite(filename, np.uint8(gcam))
        return None
    return gcam


def _preprocess(image_path, mean=None, std=None, input_size=None):
    original_image = cv2.imread(image_path)
    # original_image = cv2.resize(original_image, (270,) * 2)

    mean = (0.5, 0.5, 0.5) if mean is None else mean
    std = (0.5, 0.5, 0.5) if std is None else std
    input_size = (32,) if input_size is None else input_size

    raw_image = cv2.resize(original_image, input_size * 2)
    image = transforms.Compose(
        [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        ]
    )(raw_image[..., ::-1].copy())
    return image, original_image


def load_images(image, **kwargs):

    mean = kwargs['mean'] if 'mean' in kwargs else None
    std = kwargs['std'] if 'std' in kwargs else None
    input_size = kwargs['input_size'] if 'input_size' in kwargs else None

    images = []
    raw_images = []
    image, raw_image = _preprocess(image, mean, std, input_size) if type(image) is str else (image, image.numpy()[::-1])
    images.append(image)
    raw_images.append(raw_image)
    return images, raw_images


