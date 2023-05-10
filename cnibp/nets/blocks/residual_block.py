import torch
import torch.nn as nn


class ResBlock_1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=2, idx=1):
        super(ResBlock_1D, self).__init__()
        self.out_channels = idx * out_channels
        self.dilation = idx * dilation
        # self.conv1 = nn.Sequential(
        #     nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
        #               kernel_size=kernel_size, stride=stride, dilation=self.dilation, padding=self.dilation),
        #     nn.BatchNorm1d(out_channels)
        # )
        # self.conv2 = nn.Sequential(
        #     nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
        #                 kernel_size=kernel_size, stride=stride, dilation=self.dilation, padding=self.dilation),
        #     nn.BatchNorm1d(out_channels)
        # )
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=self.dilation, dilation=self.dilation)
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=self.dilation, dilation=self.dilation)
        self.channel_expansion = nn.Conv1d(in_channels=out_channels, out_channels=out_channels*2, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.batch_norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):

        residual = x
        out1 = self.batch_norm(self.conv1(x))
        out2 = self.relu(out1)
        out3 = self.batch_norm(self.conv2(out2))
        out4 = out3 + residual
        out5 = self.relu(out4)

        _, channelN, _ = out5.shape

        return self.channel_expansion(out5), channelN
