import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import numpy as np
import pandas as pd
import torch.optim as optim
import gc
from torch.utils.data import DataLoader, Dataset
from component import TCN, BasicBlock, ChannelAttention, conv3x3, eca_layer
import torch.nn.functional as F

class MyBlockWithVMD(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(MyBlockWithVMD, self).__init__()
        # self.BasicBlock = BasicBlock(inplanes, planes, stride=1, downsample=None)
        # self.BatchNorm1d = nn.BatchNorm1d(120, affine=False)
        self.ca = eca_layer(12)
        # self.gru1 = nn.GRU(1, 2, 1, batch_first = True)
        self.tcn = TCN(120, [90,60,30,10])
        self.fc1 = nn.Linear(336, 50)
        # self.fc2 = nn.Linear(240, 20)
        self.fc3 = nn.Linear(100, 1)
        self.relu = nn.ReLU()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv = nn.Conv2d(12, 12, kernel_size=(10,1), stride=1,
                     padding=0, bias=False)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x:10000 x 120 x 24
        # x = self.BatchNorm1d(x)
        batch_size = x.shape[0]
        outwater = x[:, :, -5:]
        # h0 = torch.randn(1, batch_size, 2).to(device)
        outwater = outwater.reshape(batch_size, -1, 5)
        
        # out1, _ = self.gru1(outwater, h0)
        out1 = self.tcn(outwater)
        out1 = F.relu(out1.reshape(batch_size, -1))
        

        x2 = x.reshape(batch_size, 12, 10, 28)

        out = self.conv1(x2)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        out2 = self.ca(out) * out
        # out2 = x2.mean(2)
        out2 = self.conv(out2)
        out2 = out2.reshape(batch_size, -1)   
        
        # out2 = self.BasicBlock(out2)

        out2 = self.fc1(out2)
        out = torch.cat([out1, out2],dim=1)

        out = F.relu(self.dropout(self.fc3(out)))
        
        return out