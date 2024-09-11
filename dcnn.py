import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

class DCNN(nn.Module):
    def __init__(self, in_channels):
        super(DCNN, self).__init__()
        
        self.in_channels = in_channels

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(8, 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.predict = nn.Sequential(
            nn.Dropout(0.1),
            nn.ReLU(inplace=True),
            nn.Linear(112, 1),
        )
 
    def forward(self, x):
        # x: 10000 x 12 x 10 x24
        batch_size = x.shape[0]
        x = self.features(x)
        x = x.reshape(batch_size, -1)
        x = self.predict(x)
        return x
