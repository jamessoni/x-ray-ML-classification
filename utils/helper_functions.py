"""
Function to help load and visualise our data
"""

import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedShuffleSplit


def get_data(filepath, transform=None, rgb=False):
    """
    Read in data from the given folder.

    Parameters
    ----------
    filepath: str
        Path for the file e.g.: F'string/containing/filepath'
    transform: callable
        A function which tranforms the data to the required format
    rgb: bool
        Image type for different channel types

    Returns
    -------
    torchvision.datasets.folder.ImageFolder
        Required data after read in

    """
    if transform is None:
        transform = transforms.Compose([transforms.ToTensor()])

    if rgb:
        # will read the data into 3 channels,
        # majority of images are 1 channel only however
        xray_data = datasets.ImageFolder(root=filepath,
                                         transform=transform)

    else:
        # read in all images as one channel
        xray_data = datasets.ImageFolder(root=filepath, transform=transform)

    return xray_data


def visualise_rand(data, seed=42, n=3):
    """
    Visualise a set of our images to see if images imported correctly.

    Parameters
    ----------
    data: torchvision.datasets.folder.ImageFolder
        Data to visualise
    seed: int, optinional
        Seed for the random generator.
    n: int, optional
        Number of image in a row

    Returns
    -------
    None
        The result is plotted and not stored

    """
    set_seed(seed)

    plt.rcParams["figure.figsize"] = (20, 20)   # figure sizes
    plt.figure(0)
    for i in range(n):
        for j in range(n):
            sample_num = np.random.randint(
                len(data.targets))   # pick random image
            sample, target = data[sample_num]
            ax = plt.subplot2grid((5, 5), (i, j))  # subplot
            ax.imshow(sample, cmap='gray')
            ax.set_title(
                data.classes[data.targets[sample_num]])  # obtain class label
    plt.show()


def set_seed(seed):
    """
    Use this to set ALL the random seeds to a fixed value
    and take out any randomness from cuda kernels.

    Parameters
    ----------
    seed: int
        Seed for the random generators

    Returns
    -------
    Bool
        Indicator whether the seed setting was successful

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True
    # uses the inbuilt cudnn auto-tuner to find the
    # fastest convolution algorithms.
    torch.backends.cudnn.enabled = True
    return True


def initialize_data_norm(filepath, filepath_test,
                         mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """
    Load three whole datasets(train, validate and test)
    and apply normalization on test dataset.

    Parameters
    ----------
    filepath: str
        Path for the file e.g.: F'string/containing/filepath'
    filepath_test: str
        Path for the test file e.g.: F'string/containing/filepath/test'

    Returns
    -------
    tuple
        3 element-long tuple containing training dataset, validation dataset
            and testing dataset

    """
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        # transforms.Normalize(
        #     (0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))

    ])
    xray_train = get_data(filepath, transform=None)
    xray_valid = get_data(filepath, transform=None)
    xray_test = get_data(filepath_test, transform=test_transform)

    return xray_train, xray_valid, xray_test


def get_train_valid_data(filepath, filepath_test, test_size,
                         mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """
    Get the datasets for train and validation without data augmentation.

    Parameters
    ----------
    filepath: str
        Path for the file e.g.: F'string/containing/filepath'
    filepath_test: str
        Path for the test file e.g.: F'string/containing/filepath/test'
    test_size: int
        Size of test data

    Returns
    -------
    tuple
        3 element-long tuple containing training dataset, validation dataset
            and testing dataset

    """
    # This is the transform without data augmentation
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        # transforms.Normalize(
        #     (0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))

    ])
    xray_train, xray_valid, xray_test = initialize_data_norm(
        filepath, filepath_test)
    labels = xray_train.targets
    shuffler = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=42)
    train_indices, valid_indices = list(
        shuffler.split(np.array(labels)[:, np.newaxis], labels))[0]
    trainset = torch.utils.data.Subset(xray_train, train_indices)
    validset = torch.utils.data.Subset(xray_valid, valid_indices)
    trainset.dataset.transform = train_transform
    validset.dataset.transform = train_transform

    return trainset, validset, xray_test


def get_data_augmented(filepath, filepath_test, test_size,
                       mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """
    Get the datasets for train and validation with data augmentation.

    Parameters
    ----------
    filepath: str
        Path for the file e.g.: F'string/containing/filepath'
    filepath_test: str
        Path for the test file e.g.: F'string/containing/filepath/test'
    test_size: int
        Size of test data

    Returns
    -------
    tuple
        3 element-long tuple containing training dataset, validation dataset
            and testing dataset with data augmentation

    """
    # This is the transform for data augmentation
    augmentation_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.ColorJitter(contrast=1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        # transforms.Normalize(
        #     (0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
    ])

    xray_train, xray_valid, xray_test = initialize_data_norm(
        filepath, filepath_test, mean=mean, std=std)
    labels = xray_train.targets
    shuffler = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=42)
    train_indices, valid_indices = list(
        shuffler.split(np.array(labels)[:, np.newaxis], labels))[0]
    trainset = torch.utils.data.Subset(xray_train, train_indices)
    validset = torch.utils.data.Subset(xray_valid, valid_indices)
    trainset.dataset.transform = augmentation_transform
    validset.dataset.transform = train_transform

    return trainset, validset, xray_test
