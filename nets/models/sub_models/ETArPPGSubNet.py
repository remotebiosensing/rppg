# import pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from nets.blocks.ETArPPGBlocks import ETASubNetBlock


# define ETA-rPPGNet SubNet
class ETArPPGNetSubNet(nn.Module):
    def __init__(self, shape):
        # segment shape : (C, t, W, H)
        [self.c, self.t, self.h, self.w] = shape
        super(ETArPPGNetSubNet, self).__init__()
        self.h //= 2  # output height
        self.w //= 2  # output width
        self.ETASubNetBlock = ETASubNetBlock()

    def forward(self, x):
        # Input Shape : (N, C, t, H, W)
        # Output Shape : (N, C, H/2, W/2)
        featuremap = torch.zeros((x.shape[0], self.c, self.h, self.w))

        # transform segment to feature map
        for i in range(x.shape[0]):
            # concat feature map
            featuremap[i, :, :, :] = self.ETASubNetBlock(x[i])

        return featuremap
