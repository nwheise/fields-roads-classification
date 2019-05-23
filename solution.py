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

TEST_SPLIT = 0.2
SHUFFLE_DATASET = True
RANDOM_SEED = 73
BATCH_SIZE = 2
EPOCHS = 2


def get_device():
    '''
    Returns cuda device if it is available, else cpu
    '''

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    return device


def produce_data_loaders(data_folder, transform, TEST_SPLIT,
                         shuffle_dataset, batch_size):
    '''
    Returns a train loader and test loader for image data contained in
    data_folder. Images should be located in folders according to their class.

    Parameters
    data_folder: folder containing images (placed in subfolders)
    transform: PyTorch transforms to be performed on images
    TEST_SPLIT: [0, 1] proportion of data for test
    shuffle_dataset: boolean, True if data should be shuffled before split
    batch_size: batch size for data loaders
    '''

    dataset = datasets.ImageFolder(data_folder, transform=transform)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(TEST_SPLIT * dataset_size))
    if shuffle_dataset:
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(indices)
    val_indices, train_indices  = indices[:split], indices[split:]

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset,
                                                    batch_size=batch_size,
                                                    sampler=test_sampler)

    return train_loader, test_loader


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
        transforms.Resize(size=50),
        transforms.RandomResizedCrop(size=50),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        ])

    train_loader, test_loader = produce_data_loaders(DATA_FOLDER,
                                                           transform,
                                                           TEST_SPLIT,
                                                           SHUFFLE_DATASET,
                                                           BATCH_SIZE)

    device = get_device()

    if False:
        save_images_from_loader(data_loader=train_loader,
                                folder=os.path.join(TRANSFORMS_FOLDER, 'train'))
        save_images_from_loader(data_loader=test_loader,
                                folder=os.path.join(TRANSFORMS_FOLDER, 'test'))

    net = Net().to(device)
    optimizer = torch.optim.SGD(params=net.parameters(), lr=0.001, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    print('Begin training')
    print('Epoch  |  Batch  |  Loss')
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # get inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward pass
            outputs = net(inputs)

            # backward pass
            loss = criterion(outputs, labels)
            loss.backward()

            # optimize
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            checkpoint = 10
            if i % checkpoint == checkpoint - 1:
                print(f'{epoch}  {i}  {round(running_loss / checkpoint, 4)}\t')
                running_loss = 0.0

    # Evaluate accuracy on test
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Accuracy of the net on the {total} test images: {100 * correct / total}%')




if __name__ == '__main__':
    main()