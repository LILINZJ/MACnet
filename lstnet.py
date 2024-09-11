import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTnet(nn.Module):
    def __init__(self):
        super(LSTnet, self).__init__()
        self.use_cuda = True
        self.P = 120
        self.m = 28
        self.hidR = 6
        self.hidC = 12
        self.hidS = 24
        self.Ck = 3
        self.skip = 2
        self.pt = (self.P - self.Ck) // self.skip
        self.hw = 120
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.m))
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(p=0.1)
        if self.skip > 0:
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.m)
        else:
            self.linear1 = nn.Linear(self.hidR, self.m)
        if self.hw > 0:
            self.highway = nn.Linear(self.hw, 1)
        self.linear = nn.Linear(29, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)

        # CNN
        c = x.view(-1, 1, self.P, self.m)
        c = torch.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)
        
        # RNN 
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r, 0))
        
        # skip-rnn

        if self.skip > 0:
            s = c[:, :, int(-self.pt * self.skip):].contiguous()
            s = s.view(batch_size, self.hidC, self.pt, self.skip)
            s = s.permute(2, 0, 3, 1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.hidC)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.hidS)
            s = self.dropout(s)
            r = torch.cat((r, s), 1)

        res = self.linear1(r)
      
        # high way
        if self.hw > 0:
            z = x[:, -self.hw:, 1]
            z = z.contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1, 1)
            res = torch.cat([res, z], dim=1)
        
        out = self.relu(self.linear(res))

        return out