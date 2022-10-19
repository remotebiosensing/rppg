# import pytorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from nets.blocks.ETArPPGBlocks import STBlock, TimeDomainAttention, rPPGgenerator


# define ETA-rPPGNet
class ETArPPGNet(nn.Module):
    def __init__(self, length):
        super(ETArPPGNet, self).__init__()
        self.length = length
        # define ETA-rPPGNet layers
        self.etarppgnet = nn.Sequential(
            STBlock(),
            STBlock(),
            STBlock(),
            STBlock(),
            TimeDomainAttention(),
            rPPGgenerator(self.length)
        )

    def forward(self, x):
        x = self.etarppgnet(x)
        return x.view(-1, self.length)

if __name__=='__main__':
    x = torch.randn(2, 3, 32, 128, 128)
    net = ETArPPGNet(64)
    y = net(x)
    print(x.shape)
    print(y.shape)