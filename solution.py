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
FIELDS_FOLDER = 'fields'
ROADS_FOLDER = 'roads'
TRANSFORMS_FOLDER = 'transforms'

TEST_SPLIT = 0.2
SHUFFLE_DATASET = True
RANDOM_SEED = 73
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


def produce_data_loaders(data_folder, transform):
    '''
    Returns a train loader and test loader for image data contained in
    data_folder. Images should be located in folders according to their class.

    Parameters
    data_folder: folder containing images (placed in subfolders)
    train_transform: PyTorch transforms to be performed on training images
    test_transform: PyTorch transforms to be performed on test images
    '''

    # Create dataset from images
    dataset = datasets.ImageFolder(data_folder, transform=transform)
    dataset_size = len(dataset)
    print(f'Dataset size: {dataset_size}')

    # Perform train / test split
    indices = list(range(dataset_size))
    split = int(np.floor(TEST_SPLIT * dataset_size))
    print(f'Train size: {dataset_size - split}')
    print(f'Test size: {split}')

    if SHUFFLE_DATASET:
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(indices)

    test_indices, train_indices  = indices[:split], indices[split:]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # Build data loaders from the dataset
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=BATCH_SIZE,
                                               sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=BATCH_SIZE,
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

    for i, data in enumerate(data_loader):
        for j in range(data[0].shape[0]):
            label = data[1][j]
            to_pil_img = transforms.ToPILImage()
            img = to_pil_img(data[0][j])
            img.save(os.path.join(folder, f'{i}_{j}_{label}.jpg'))


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

    # Specify transforms on original data
    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=(50, 50), scale=(0.75, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        ])

    # Produce train and test data loaders
    train_loader, test_loader = produce_data_loaders(data_folder=DATA_FOLDER,
                                                     transform=transform)

    # Optionally save the transformed images that were created from originals
    if False:
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