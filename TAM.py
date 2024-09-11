import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class TAM(nn.Module):
    def __init__(self,
                 in_channels,
                 n_segment,
                 kernel_size=3,
                 stride=1,
                 padding=1):
        super(TAM, self).__init__()
        self.in_channels = in_channels
        self.n_segment = n_segment
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.G = nn.Sequential(
            nn.Linear(n_segment, n_segment * 2, bias=False),
            nn.BatchNorm1d(n_segment * 2), nn.ReLU(inplace=True),
            nn.Linear(n_segment * 2, kernel_size, bias=False), nn.Softmax(-1))

        self.L = nn.Sequential(
            nn.Conv1d(in_channels,
                      in_channels // 4,
                      kernel_size,
                      stride=1,
                      padding=kernel_size // 2,
                      bias=False), nn.BatchNorm1d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // 4, in_channels, 1, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        # x.size = N*C*T*(H*W)
        10000*12*10*28
        nt, c, t, hw = x.size()
        c = 12
        t = 10
        h = 4
        w = 7
        n_batch = nt 
        new_x = x.view(n_batch, c, t, h, w).contiguous()
        out = F.adaptive_avg_pool2d(new_x.view(n_batch, t * c, h, w), (1, 1))
        out = out.view(-1, t)
        conv_kernel = self.G(out.view(-1, t)).view(n_batch, 1, -1, 1)
        local_activation = self.L(out.view(n_batch, c,
                                           t)).view(n_batch, c, t, 1, 1)
        new_x = new_x * local_activation
        out = F.conv2d(new_x.view(1, n_batch * c, t, h * w),
                       conv_kernel,
                       bias=None,
                       stride=(self.stride, 1),
                       padding=(self.padding, 0),
                       groups=n_batch * c)
        out = out.view(n_batch, c, t, h, w)
        out = out.permute(0, 2, 1, 3, 4).contiguous().view(nt, c, h, w)

        return out


class TemporalBottleneck(nn.Module):
    def __init__(self,
                 net,
                 n_segment=8,
                 t_kernel_size=3,
                 t_stride=1,
                 t_padding=1):
        super(TemporalBottleneck, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.tam = TAM(in_channels=net.conv1.out_channels,
                       n_segment=n_segment,
                       kernel_size=t_kernel_size,
                       stride=t_stride,
                       padding=t_padding)

    def forward(self, x):
        identity = x

        # out = self.net.conv1(x)
        # out = self.net.bn1(out)
        # out = self.net.relu(out)
        out = self.tam(x)

        out = self.net.conv2(out)
        out = self.net.bn2(out)
        out = self.net.relu(out)

        out = self.net.conv3(out)
        out = self.net.bn3(out)

        if self.net.downsample is not None:
            identity = self.net.downsample(x)

        out += identity
        out = self.net.relu(out)

        return out
