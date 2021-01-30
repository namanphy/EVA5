from torchvision import datasets, transforms


def cifar10_dataset(location='../data', train=True, download=True, transform=None):
    return datasets.CIFAR10(location, train=train, download=download, transform=transform)


def transformations(augmentation=False, rotation=0.0, randomHorizontalFlip=False):
    transforms_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    augmentation_list = []
    if randomHorizontalFlip:
        augmentation_list.append(transforms.RandomHorizontalFlip())

    if rotation != 0.0:
        augmentation_list.append(transforms.RandomRotation((-rotation, rotation), fill=(1,)))

    if augmentation:
        transforms_list = transforms_list + augmentation_list

    return transforms.Compose(transforms_list)