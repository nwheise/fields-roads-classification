import os
import torch
import torchvision
from torchvision import datasets, transforms, models, utils
from nets import Net
import matplotlib.pyplot as plt
from PIL import Image


DATA_FOLDER = 'data'
FIELDS_FOLDER = 'fields'
ROADS_FOLDER = 'roads'
TRANSFORMS_FOLDER = 'transforms'
BATCH_SIZE = 1
EPOCHS = 5

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

transform = transforms.Compose([
    transforms.Resize(size=100),
    transforms.RandomResizedCrop(size=75),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
    ])

classes = ('field', 'road')

train_data = datasets.ImageFolder(DATA_FOLDER, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(EPOCHS):
    if not os.path.isdir(os.path.join(DATA_FOLDER, TRANSFORMS_FOLDER)):
        os.makedirs(os.path.join(DATA_FOLDER, TRANSFORMS_FOLDER))

    for i, data in enumerate(train_loader):
        to_pil_img = transforms.ToPILImage()
        img = to_pil_img(data[0][0])
        img.save(os.path.join(DATA_FOLDER, TRANSFORMS_FOLDER, f'{epoch}_{i}.jpg'))