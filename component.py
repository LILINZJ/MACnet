import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import numpy as np
import pandas as pd
import torch.optim as optim
import gc
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()
        self.hardshrink = nn.Hardshrink(0.2)

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.hardshrink(self.sigmoid(y))

        return x * y.expand_as(x)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 6, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 6, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        # self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.conv1 = nn.Conv2d(2, 1, kernel_size, 1, 2, dilation = 2)
        self.sigmoid = nn.Sigmoid()
        self.hardshrink = nn.Hardshrink(0.2)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.hardshrink(self.sigmoid(x))

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        # self.ca = ChannelAttention(planes)
        self.ca = eca_layer(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # out = self.conv2(out)
        # out = self.bn2(out)
        
        out1 = self.ca(out) * out
        out2 = self.sa(out) * out

        
        if self.downsample is not None:
            residual = self.downsample(x)
       
        out1 += residual
        out2 += residual

        # out1 = self.conv1(out1)
        # out1 = self.bn1(out1)
        

        # out2 = self.conv2(out2)
        # out2 = self.bn2(out2)
        
        out1 = self.relu(out1)
        length = out1.shape[0]
        out1 = out1.contiguous().view(length,-1)
        out2 = self.relu(out2)
        out2 = out2.contiguous().view(length,-1)

        out = torch.cat([out1, out2],dim=1)

        return out

class BasicBlock1(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock1, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        # self.ca = ChannelAttention(planes)
        self.ca = eca_layer(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # out = self.conv2(out)
        # out = self.bn2(out)
        
        out1 = self.ca(out) * out
        out2 = self.sa(out) * out

        
        if self.downsample is not None:
            residual = self.downsample(x)
       
        out1 += residual
        out2 += residual

        # out1 = self.conv1(out1)
        # out1 = self.bn1(out1)
        

        # out2 = self.conv2(out2)
        # out2 = self.bn2(out2)
        
        out1 = self.relu(out1)
        length = out1.shape[0]
        out1 = out1.contiguous().view(length,-1)
        out2 = self.relu(out2)
        out2 = out2.contiguous().view(length,-1)

        out = torch.cat([out1, out2],dim=1)

        return out

class TSA(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(TSA, self).__init__()
        self.lstm1 = nn.LSTM(in_feature, out_feature, batch_first=True)
        self.lstm2 = nn.LSTM(in_feature, out_feature, batch_first=True)
        self.lstm3 = nn.LSTM(in_feature, out_feature, batch_first=True)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):

        q,_ = self.lstm1(x)
        k,_ = self.lstm2(x)
        v,_ = self.lstm3(x)
        
        a = self.softmax(q.mul(k))
        b = v.mul(a)
        
        return b


class TemporalPatternAttentionMechanism(nn.Module):
    def __init__(self, attn_size, attn_length, filter_num):
        super(TemporalPatternAttentionMechanism, self).__init__()
        self.filter_num = filter_num
        self.filter_size = 1
        self.attn_size = attn_size
        self.dense = nn.Linear(self.attn_size, self.filter_num)
        self.attn_length = attn_length
        self.conv = nn.Conv2d(in_channels=1, out_channels=self.filter_num,
                              kernel_size=(attn_length, self.filter_size),
                              padding=(0, 0), bias=False)
        self.fc1 = nn.Linear(attn_size + self.filter_num, attn_size)

    def forward(self, query, attn_states):
        """
        query: [batch_size, attn_size] (original series)
        attn_states: [batch_size, attn_length, attn_size] (VMD sub-series)
        new_attns: [batch_size, attn_size] 
        """
        batch_size = query.size(0)
        w = self.dense(query).view(-1,1,self.filter_num)

        attn_states = attn_states.view(batch_size, -1, self.attn_length, self.attn_size)
        # conv_vecs: [batch_size, feature_dim, filter_num]
        conv_vecs = self.conv(attn_states)
        conv_vecs = conv_vecs.view(batch_size, self.attn_size - self.filter_size + 1, self.filter_num)
        # s: [batch_size, feature_dim]
        s = torch.sum(torch.mul(conv_vecs, w), dim=2)
        # a: [batch_size, feature_dim]
        a = torch.sigmoid(s)
        # d: [batch_size, feature_dim, filter_num]
        d = torch.mul(a.view(batch_size, -1, 1), conv_vecs) # 广播机制

        new_conv_vec = d.view(batch_size, -1, self.filter_num)
        # new_attns = self.fc1(torch.cat([query, new_conv_vec], dim=1))
        return new_conv_vec

class TPA_TCN(nn.Module):
    def __init__(self, SeriesLen, VMD_k, filter_num, num_channels) -> None:
        super(TPA_TCN, self).__init__()
        self.tpa = TemporalPatternAttentionMechanism(attn_size=SeriesLen, attn_length=VMD_k, filter_num=filter_num)
        self.tcn = TCN(SeriesLen, num_channels, filter_num)
        self.k = VMD_k

    def forward(self, x):
        batch_size = x.shape[0]
        xVMD = x[:,:,:-1].permute(0,2,1)
        orgSer = x[:,:,-1].view(batch_size, -1)
        out = self.tpa(orgSer, xVMD)
        out = self.tcn(out)

        return out