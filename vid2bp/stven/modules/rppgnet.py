import torch
import torch.nn as nn
import torch.nn.functional as F

class rppgNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(rppgNet, self).__init__()
        self.Conv1 = nn.Conv3d(in_channels, 32, kernel_size=(1, 5, 5))
        self.batchnorm1 = nn.BatchNorm3d(32)
        self.relu1 = nn.ReLU()
        # self.ST_Block = ST_Block(32, 64, kernel_size=(3, 3, 3))
        self.SGAP = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.batch_norm2 = nn.BatchNorm3d(64)
        self.relu2 = nn.ReLU()
        self.Conv_2 = nn.Conv3d(64, 1, kernel_size=(1, 1, 1))
        self.batch_norm3 = nn.BatchNorm3d(1)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.Conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        # x = self.ST_Block(x)
        x = self.SGAP(x)
        x = self.batch_norm2(x)
        x = self.relu2(x)
        x = self.Conv_2(x)
        x = self.batch_norm3(x)
        x = self.relu3(x)
        return x
