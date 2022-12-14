import torch
import torch.nn as nn


class stven(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(stven, self).__init__()
        self.Conv_1 = nn.Conv3d(in_channels, 64, kernel_size=(3, 7, 7))  # output: (64, T, 128, 128)
        self.Conv_2 = nn.Conv3d(64, 128, kernel_size=(3, 4, 4))  # output: (128, T, 64, 64)
        self.Conv_3 = nn.Conv3d(128, 512, kernel_size=(4, 4, 4))  # output: (512, T/2, 32, 32)
        # self.ST_Block = ST_Block(512, 512, kernel_size=(3, 3, 3))
        self.DConv_1 = nn.ConvTranspose3d(512, 128, kernel_size=(4, 4, 4))  # output: (128, T, 64, 64)
        self.DConv_2 = nn.ConvTranspose3d(128, 64, kernel_size=(1, 4, 4))  # output: (64, T, 128, 128)
        self.DConv_3 = nn.ConvTranspose3d(64, 3, kernel_size=(1, 7, 7))  # output: (3, T, 128, 128)

    def forward(self, x):
        x = self.Conv_1(x)
        x = self.Conv_2(x)
        x = self.Conv_3(x)
        # x = self.ST_Block(x)
        x = self.DConv_1(x)
        x = self.DConv_2(x)
        x = self.DConv_3(x)
        return x
