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
            nn.MaxPool1d(kernel_size=2))

        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2))

        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 256, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2))

        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, 512, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2))

        self.conv5 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Conv1d(1024, 1024, kernel_size=3, padding='same'),
            nn.ReLU())

        self.level4 = nn.Conv1d(1024, 1, kernel_size=1)  # input: conv5

        self.up6 = nn.functional.upsample(1024, scale_factor=2)  # input : conv5

        # after concat up6, conv4 : 1024 + 512 = 1536

        self.conv6 = nn.Sequential(
            nn.Conv1d(1536, 512, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, 512, kernel_size=3, padding='same'),
            nn.ReLU())

        self.level3 = nn.Conv1d(512, 1, kernel_size=1)  # input: conv6

        self.up7 = nn.functional.upsample(512, scale_factor=2)  # input : conv6

        # after concat up7,conv3 : 512 + 256 = 768

        self.conv7 = nn.Sequential(
            nn.Conv1d(768, 256, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 256, kernel_size=3, padding='same'),
            nn.ReLU())

        self.level2 = nn.Conv1d(256, 1, kernel_size=1)  # input: conv7

        self.up8 = nn.functional.upsample(256, scale_factor=2)  # input : conv7

        # after concat up8,conv2 : 256 + 128 = 384

        self.conv8 = nn.Sequential(
            nn.Conv1d(384, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=3, padding='same'),
            nn.ReLU())

        self.level1 = nn.Conv1d(128, 1, kernel_size=1)  # input: conv8

        self.up9 = nn.functional.upsample(128, scale_factor=2)  # input : conv8

        # after concat up9,conv1 : 128 + 64 = 192

        self.conv9 = nn.Sequential(
            nn.Conv1d(192, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, kernel_size=3, padding='same'),
            nn.ReLU())

        self.out = nn.Conv1d(64, 1, kernel_size=1)  # input: conv9

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        level4 = self.level4(conv5)
        up6 = self.up6(conv5)
        concat6 = torch.cat([up6, conv4], dim=1)
        conv6 = self.conv6(concat6)
        level3 = self.level3(conv6)
        up7 = self.up7(conv6)
        concat7 = torch.cat([up7, conv3], dim=1)
        conv7 = self.conv7(concat7)
        level2 = self.level2(conv7)
        up8 = self.up8(conv7)
        concat8 = torch.cat([up8, conv2], dim=1)
        conv8 = self.conv8(concat8)
        level1 = self.level1(conv8)
        up9 = self.up9(conv8)
        concat9 = torch.cat([up9, conv1], dim=1)
        conv9 = self.conv9(concat9)
        out = self.out(conv9)

        return out, level1, level2, level3, level4



