import torch
import torch.nn as nn

"""
Check Lists:
1. concat dim
"""


class UNetDS64(nn.Module):
    """
        Deeply supervised U-Net with kernels multiples of 64

    Arguments:
        length {int} -- length of the input signal

    Keyword Arguments:
        n_channel {int} -- number of channels in the output (default: {1})

    """

    def __init__(self, length, n_channel=1):
        super(UNetDS64, self).__init__()
        self.length = length
        self.n_channel = n_channel

        self.conv1 = nn.Sequential(
            nn.Conv1d(n_channel, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(64))

        self.pool1 = nn.MaxPool1d(kernel_size=2)  # input (N, 1, Length) output (N, 64, Length/2)

        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(128))

        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 256, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(256))

        self.pool3 = nn.MaxPool1d(kernel_size=2)

        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, 512, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(512))

        self.pool4 = nn.MaxPool1d(kernel_size=2)

        self.conv5 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Conv1d(1024, 1024, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(1024))

        self.level4 = nn.Conv1d(1024, 1, kernel_size=1, padding='same')

        self.up6 = nn.Upsample(scale_factor=2)

        self.conv6 = nn.Sequential(
            nn.Conv1d(1536, 512, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, 512, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(512))

        self.level3 = nn.Conv1d(512, 1, kernel_size=1)  # input: conv6

        self.up7 = nn.Upsample(scale_factor=2)  # input : conv6

        self.conv7 = nn.Sequential(
            nn.Conv1d(768, 256, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 256, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(256))

        self.level2 = nn.Conv1d(256, 1, kernel_size=1)  # input: conv7

        self.up8 = nn.Upsample(scale_factor=2)  # input : conv7

        self.conv8 = nn.Sequential(
            nn.Conv1d(384, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(128))

        self.level1 = nn.Conv1d(128, 1, kernel_size=1)  # input: conv8

        self.up9 = nn.Upsample(scale_factor=2)  # input : conv8

        self.conv9 = nn.Sequential(
            nn.Conv1d(192, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(64))

        self.out = nn.Conv1d(64, 1, kernel_size=1)  # input: conv9

    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        conv5 = self.conv5(pool4)
        level4 = self.level4(conv5)

        up6 = self.up6(conv5)
        merge6 = torch.cat([conv4, up6], dim=1)
        conv6 = self.conv6(merge6)
        level3 = self.level3(conv6)

        up7 = self.up7(conv6)
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7 = self.conv7(merge7)
        level2 = self.level2(conv7)
        up8 = self.up8(conv7)
        merge8 = torch.cat([conv2, up8], dim=1)
        conv8 = self.conv8(merge8)
        level1 = self.level1(conv8)
        up9 = self.up9(conv8)
        merge9 = torch.cat([conv1, up9], dim=1)
        conv9 = self.conv9(merge9)
        out = self.out(conv9)
        # out = torch.cat([out, level1, level2, level3, level4], dim=-1)

        return out, level1, level2, level3, level4
