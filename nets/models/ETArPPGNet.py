# import pytorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from nets.blocks.ETArPPGBlocks import STBlock, TimeDomainAttention, rPPGgenerator

# define ETA-rPPGNet
class ETArPPGNet(nn.Module):
    def __init__(self, blocks, length):
        super(ETArPPGNet, self).__init__()
        self.length = length
        self.blocks = blocks
        # define ETA-rPPGNet layers
        self.etarppgnet = nn.Sequential(
            STBlock(),
            STBlock(),
            STBlock(),
            STBlock(),
            TimeDomainAttention(),
            rPPGgenerator(self.blocks, self.length)
        )

    def forward(self, x):
        x = self.etarppgnet(x)
        return x.view(self.length, -1)

