from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import typing
import numpy as np
import pathlib
np.random.seed(0)

mean = (0.5, 0.5, 0.5)
std = (.25, .25, .25)


def get_data_dir():
    server_dir = pathlib.Path("/work/datasets/cifar10")
    if server_dir.is_dir():
        return str(server_dir)
    return "data/cifar10"


def load_cifar10(
    batch_size: int,
    transform_train_val=None,
    transform_test=None,
    validation_fraction: float = 0.1
) -> typing.List[torch.utils.data.DataLoader]:
    """
    Args:
        batch_size (int)
        transform_train_val (optional): The transformations to perform on the train and validation data
        transform_test (optional): The transformations to perform on the test data

    Returns:
        typing.List[torch.utils.data.DataLoader]: train-, val- and test-dataloader
    """
    # Will be used unless the dataloader is explicitly given a transform
    norm_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    if transform_train_val is None:
        transform_train_val = norm_transform
    if transform_test is None:
        transform_test = norm_transform

    data_train = datasets.CIFAR10(
        get_data_dir(),
        train=True,
        download=True,
        transform=transform_train_val,
    )

    data_test = datasets.CIFAR10(
        get_data_dir(),
        train=False,
        download=True,
        transform=transform_test,
    )

    indices = list(range(len(data_train)))
    split_idx = int(np.floor(validation_fraction * len(data_train)))

    val_indices = np.random.choice(indices, size=split_idx, replace=False)
    train_indices = list(set(indices) - set(val_indices))

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)

    dataloader_train = torch.utils.data.DataLoader(
        data_train,
        batch_size=batch_size,
        num_workers=2,
        sampler=train_sampler,
        drop_last=True,
    )
    dataloader_val = torch.utils.data.DataLoader(
        data_train,
        batch_size=batch_size,
        num_workers=2,
        sampler=validation_sampler,
    )
    dataloader_test = torch.utils.data.DataLoader(
        data_test,
        batch_size=batch_size,
        num_workers=2,
        shuffle=False,
    )
    return dataloader_train, dataloader_val, dataloader_test
