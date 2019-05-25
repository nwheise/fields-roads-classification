import torch
import torch.nn as nn
import torch.nn.functional as F

class FieldRoadNet(nn.Module):

    def __init__(self):
        super(FieldRoadNet, self).__init__()

        # 50x50 input
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5) # 46x46
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2)) # 23x23
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3) # 21x21
        # another maxpool makes it 10x10
        self.fc1 = nn.Linear(in_features=12*10*10, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool(x)

        _, C, H, W = x.data.size()
        x = x.view(-1, C*H*W)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        x = torch.sigmoid(x)

        return x
