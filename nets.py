import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # 50x50 input
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5) # 46x46
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2)) # 24x24
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5) # 20x20
        # another maxpool makes it 10x10
        self.fc1 = nn.Linear(in_channels=12*10*10, out_channels=200)
        self.fc2 = nn.Linear(in_channels=200, out_channels=50)
        self.fc3 = nn.Linear(in_channels=50, out_channels=2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x.view(-1, 12*10*10)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x
