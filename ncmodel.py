import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import numpy as np
import pandas as pd
import torch.optim as optim
import gc
import torch.nn.functional as F
from component import TSA

class NCmodel(nn.Module):

    def __init__(self):
        super(NCmodel, self).__init__()
        self.conv1 = nn.Conv2d(120, 120, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(120, 120, kernel_size=(2,4), stride=1, padding=0)
        self.TSA1 = TSA(6,2)
        self.TSA2 = TSA(2,2)
        self.TSA3 = TSA(5,2)
        self.lstm1 = nn.LSTM(2, 2, batch_first=True)
        
        self.linear1 = nn.Linear(120*4,1)
        # self.linear2 = nn.Linear(128,1)
        

    def forward(self, x):
        # x:10000 x 120 x 28
        targets = x[:,:,-5:]
        timestep = x.shape[1]
        x = x.reshape(-1, timestep, 7, 4)
        out = self.conv1(x)
        out = self.conv2(out)
        
        out = out.reshape(-1, timestep, 6)
        out = self.TSA1(out)
        out,_ = self.lstm1(out)
        
        out = self.TSA2(out)
        
        out2 = self.TSA3(targets)  
        out = torch.cat([out,out2],dim=2)
        out = out.reshape(-1, 120*4)
        out = self.linear1(out)
        out = F.relu(out)
        
        return out