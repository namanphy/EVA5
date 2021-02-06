import cv2
import torch
import matplotlib.cm as cm
import numpy as np
from gradcam.gradcam import GradCAM
from cuda import enable_cuda
from torchvision import transforms


def _save_gradcam(filename, gcam, raw_image):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    cv2.imwrite(filename, np.uint8(gcam))


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


def _load_images(image_path, **kwargs):

    mean = kwargs['mean'] if 'mean' in kwargs else None
    std = kwargs['std'] if 'std' in kwargs else None
    input_size = kwargs['input_size'] if 'input_size' in kwargs else None

    images = []
    raw_images = []
    image, raw_image = _preprocess(image_path, mean, std, input_size)
    images.append(image)
    raw_images.append(raw_image)
    return images, raw_images


def plot_gradcam(image_path, model, path_to_model, layer, classes=None, **kwargs):
    device = enable_cuda()

    image, original_image = _load_images(image_path, **kwargs)
    image = torch.stack(image).to(device)

    model.load_state_dict(torch.load(path_to_model))
    model.to(device)
    model.eval()

    gcam = GradCAM(model=model, candidate_layers=layer)
    probs, ids = gcam.forward(image, original_image[0])

    gcam.backward(ids=ids[:, [0]])
    region = gcam.generate(target_layer=layer)

    print(f"\t #GRADCAM: {classes[ids[0, 0]] if classes else ids[0, 0]} ({probs[0, 0]})")

    # Grad-CAM save
    _save_gradcam(
        filename=f"gradcam-{model.__class__.__name__}-{layer}-{classes[ids[0, 0]] if classes else ids[0, 0]}.png",
        gcam=region[0, 0],
        raw_image=original_image[0]
    )

    gcam.remove_hook()
