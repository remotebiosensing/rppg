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
        self.ETASubNetBlock = ETASubNetBlock((self.h, self.w))

    def forward(self, x):
        # segment set shape : (N, C, t, W, H)
        # segment shape : (C, t, W, H)
        # feature maps before transpose : (N, C, H, W)
        featuremap = torch.zeros((x.shape[0], self.c, self.h, self.w))

        # transform segment to feature map
        # x : C, t, W, H -> t, C, H, W for convolution
        x = x.permute(1, 0, 3, 2)
        for i in range(x.shape[0]):
            # concat feature map
            featuremap[i, :, :, :] = self.ETASubNetBlock(x[i])

        # feature maps after transpose : (N, C, W, H)
        featuremap = featuremap.permute(0, 1, 3, 2)
        return featuremap
