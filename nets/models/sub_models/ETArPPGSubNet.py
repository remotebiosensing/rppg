# import pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from nets.blocks.ETArPPGBlocks import ETASubNetBlock


# define ETA-rPPGNet SubNet
class ETArPPGSubNet(nn.Module):
    def __init__(self):
        super(ETArPPGSubNet, self).__init__()
        self.ETASubNetBlock = ETASubNetBlock()

    def forward(self, x):
        # Input Shape : (N, Block, C, t, H, W)
        # Output Shape : (N, C, Block, H/2, W/2)

        [N, Block, C, t, H, W] = x.shape
        featuremap = torch.zeros((N, C, Block, H // 2, W // 2))

        # transform segment to feature map
        for i in range(Block):
            # concat feature map
            featuremap[:, :, i, :, :] = self.ETASubNetBlock(x[:, i]).view(-1, C, H // 2, W // 2)

        return featuremap
