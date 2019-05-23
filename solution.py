import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torchvision import datasets, transforms, models, utils

from nets import Net


DATA_FOLDER = 'data'
FIELDS_FOLDER = 'fields'
ROADS_FOLDER = 'roads'
TRANSFORMS_FOLDER = 'transforms'

VALIDATION_SPLIT = 0.2
SHUFFLE_DATASET = True
RANDOM_SEED = 73
BATCH_SIZE = 5
EPOCHS = 5


def get_device():
    '''
    Returns cuda device if it is available, else cpu
    '''

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    return device


def produce_data_loaders(data_folder, transform, validation_split,
                         shuffle_dataset, batch_size):
    '''
    Returns a train loader and validation loader for image data contained in
    data_folder. Images should be placed in folders according to their class.

    Parameters
    data_folder: folder containing images (placed in subfolders)
    transform: PyTorch transforms to be performed on images
    validation_split: [0, 1] proportion of data for validation
    shuffle_dataset: boolean, True if data should be shuffled before split
    batch_size: batch size for data loaders
    '''

    dataset = datasets.ImageFolder(data_folder, transform=transform)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(indices)
    val_indices, train_indices  = indices[:split], indices[split:]

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset,
                                                    batch_size=batch_size,
                                                    sampler=validation_sampler)

    return train_loader, validation_loader


def save_images_from_loader(data_loader, folder):
    '''
    Stores images from data_loader into folder as jpgs.

    Parameters
    data_loader: data loader containing image data
    folder: destination folder in which images are stored
    '''
    
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

    for epoch in range(EPOCHS):

        for i, data in enumerate(data_loader):
            # data[0] is the data, data[1] are the labels
            for j in range(data[0].shape[0]):
                label = data[1][j]
                to_pil_img = transforms.ToPILImage()
                img = to_pil_img(data[0][j])
                img.save(os.path.join(folder, f'{epoch}_{i}_{j}_{label}.jpg'))


def main():

    classes = ('field', 'road')
    transform = transforms.Compose([
        transforms.Resize(size=75),
        transforms.RandomResizedCrop(size=50),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        ])

    train_loader, validation_loader = produce_data_loaders(DATA_FOLDER,
                                                           transform,
                                                           VALIDATION_SPLIT,
                                                           SHUFFLE_DATASET,
                                                           BATCH_SIZE)

    save_images_from_loader(data_loader=train_loader,
                            folder=os.path.join(TRANSFORMS_FOLDER, 'train'))
    save_images_from_loader(data_loader=validation_loader,
                            folder=os.path.join(TRANSFORMS_FOLDER, 'validation'))


if __name__ == '__main__':
    main()