import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import split_folders

import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torchvision import datasets, transforms, models, utils

from nets import FieldRoadNet


DATA_FOLDER = 'data'
SPLIT_DATA_FOLDER = 'split_data'
TRAIN_FOLDER = 'train'
TEST_FOLDER = 'val'
TRANSFORMS_FOLDER = 'transforms'

TEST_SPLIT = 0.2
BATCH_SIZE = 5
EPOCHS = 100


def get_device():
    '''
    Returns cuda device if it is available, else cpu
    '''

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    return device


def produce_data_loader(data_folder, transform):
    '''
    Returns a data loader image data contained in data_folder.
    Images should be located in folders according to their class.

    Parameters
    data_folder: folder containing images (placed in subfolders)
    transform: PyTorch transforms to be performed on images
    '''

    # Create dataset from images
    dataset = datasets.ImageFolder(data_folder, transform=transform)
    dataset_size = len(dataset)
    print(f'Dataset size: {dataset_size}')

    # Build data loaders from the dataset
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)

    return data_loader


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

    for i, data in enumerate(data_loader):
        for j in range(data[0].shape[0]):
            label = data[1][j]
            to_pil_img = transforms.ToPILImage()
            img = to_pil_img(data[0][j])
            img.save(os.path.join(folder, f'batch{i}_img{j}_lab{label}.jpg'))


def train_network(net, optimizer, criterion, loader):
    '''
    Perform training of net on data from loader, using specified optimizer and
    criterion parameters.

    Parameters
    net: neural net to be trained
    optimizer: torch.optim (such as Adam, SGD, etc.)
    criterion: torch.nn criterion (e.g. Cross Entropy Loss)
    loader: torch.utils.data.DataLoader containing data
    '''

    device = get_device()

    print('----- Begin training -----')
    print('  Epoch  |  Avg Loss  ')
    epoch_loss = []
    for epoch in range(EPOCHS):
        running_loss = 0.0
        batches = 0
        for i, batch in enumerate(loader, 0):
            # get inputs
            inputs, labels = batch
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

            # track loss
            running_loss += loss.item()
            batches += 1

        # Print average loss for the epoch for the user
        avg_epoch_loss = round(running_loss / batches, 4)
        print(f'{epoch}'.center(9) + '|' + f'{avg_epoch_loss}'.center(12))
        epoch_loss.append(avg_epoch_loss)

    print('----- Training Done -----')

    # Create plot of loss
    plot_and_save_data(x=range(len(epoch_loss)),
                       y=epoch_loss,
                       title='Average Loss per Epoch',
                       x_lab='Epoch',
                       y_lab='Average Loss',
                       filename='loss.jpg')

    return net


def plot_and_save_data(x, y, title, x_lab, y_lab, filename):
    '''
    Generate plot from given data, and save the image.
    '''

    fig = plt.figure()
    plt.plot(x, y)
    fig.suptitle(title)
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.show()
    fig.savefig(filename)


def test_network(net, loader):
    '''
    Test a trained neural net on data from loader. Print accuracy.

    Parameters
    net: trained neural network
    loader: torch.utils.data.DataLoader containing data
    '''

    device = get_device()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Get softmax outputs
            outputs = net(inputs)
            # Predictions are the max softmax output
            _, predicted = torch.max(outputs.data, 1)

            # Add to total and correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = round(100 * correct / total, 4)
        print(f'Accuracy of the net on the {total} test images: {acc} %')


def main():

    # Split data into train/test folders
    split_folders.ratio(DATA_FOLDER, output=SPLIT_DATA_FOLDER, ratio=(0.8, 0.2))

    # Specify transforms on original data
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=(50, 50), scale=(0.6, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        ])
    test_transform = transforms.Compose([
        transforms.Resize(size=(50, 50)),
        transforms.ToTensor()
        ])

    # Produce train and test data loaders
    train_folder = os.path.join(SPLIT_DATA_FOLDER, TRAIN_FOLDER)
    train_loader = produce_data_loader(data_folder=train_folder,
                                       transform=train_transform)

    test_folder = os.path.join(SPLIT_DATA_FOLDER, TEST_FOLDER)
    test_loader = produce_data_loader(data_folder=test_folder,
                                       transform=test_transform)


    # Optionally save the transformed images that were created from originals
    if True:
        save_images_from_loader(data_loader=train_loader,
                                folder=os.path.join(TRANSFORMS_FOLDER, 'train'))
        save_images_from_loader(data_loader=test_loader,
                                folder=os.path.join(TRANSFORMS_FOLDER, 'test'))

    # Initialize objects for the net
    device = get_device()
    field_road_net = FieldRoadNet().to(device)
    optimizer = torch.optim.Adam(params=field_road_net.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    # Train the network
    field_road_net = train_network(net=field_road_net, optimizer=optimizer,
                                   criterion=criterion, loader=train_loader)

    # Evaluate accuracy on test
    test_network(net=field_road_net, loader=test_loader)


if __name__ == '__main__':
    main()