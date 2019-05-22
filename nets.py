import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(MappingNet, self).__init__()
        self.fc1 = nn.Linear(in_features=2, out_features=2)


    def forward(self, x):
        x = self.fc1(x)
        return x
