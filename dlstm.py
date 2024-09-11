import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

class DLSTM(nn.Module):
    def __init__(self):
        super(DLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=28, hidden_size=12, num_layers=3)
        self.fc1 = nn.Linear(120*12, 128)
        self.fc2 = nn.Linear(128, 1)
       
    def forward(self, x):
        # x: 10000 x 120 x24
        batch_size = x.shape[0]
        hidden = None
        out, hidden = self.lstm(x, hidden)  
               
        out = out.reshape(batch_size, -1)
        x = self.fc1(out)
        x = F.relu(x)
        x = self.fc2(x)
        return x