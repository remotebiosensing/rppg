import torch
import torch.nn as nn


class Unet(nn.Module):
    def __init__(self, in_channels):
        super(Unet, self).__init__()
        self.in_channel = in_channels

        self.maxpool = nn.MaxPool1d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3_1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same'),
            nn.LeakyReLU())  # first conv layer output_shape
        self.conv3_2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same'),
            nn.LeakyReLU(), )
        self.conv3_3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding='same'),
            nn.LeakyReLU(), )
        self.conv3_4 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.Dropout1d(0.5))
        self.conv3_5 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.Dropout1d(0.5))
        self.conv2_1 = nn.Sequential(
            nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=2, stride=1, padding='same'),
            nn.LeakyReLU())
        self.conv3_6 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding='same'),
            nn.LeakyReLU())
        self.conv2_2 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=2, stride=1, padding='same'),
            nn.LeakyReLU())
        self.conv3_7 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding='same'),
            nn.LeakyReLU())
        self.conv2_3 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=2, stride=1, padding='same'),
            nn.LeakyReLU())
        self.conv3_8 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same'),
            nn.LeakyReLU())
        self.conv2_4 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=2, stride=1, padding='same'),
            nn.LeakyReLU())
        self.conv3_9 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same'),
            nn.LeakyReLU())
        self.conv3_10 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=2, kernel_size=3, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding='same'),
            nn.LeakyReLU())

    def forward(self, ple_input):
        x = torch.reshape(ple_input, (-1, self.in_channel, 256))  # [1,256]
        x1 = self.conv3_1(x)  # [64,256]
        x2 = self.conv3_2(self.maxpool(x1))  # [128,128]
        x3 = self.conv3_3(self.maxpool(x2))  # [256,64]
        x4 = self.conv3_4(self.maxpool(x3))  # [512,32]
        x5 = self.conv3_5(self.maxpool(x4))  # [1024,16]
        x6 = self.conv2_1(self.upsample(x5))  # [512,32]
        x7 = self.conv3_6(x4 + x6)  # [512,32]
        x8 = self.conv2_2(self.upsample(x7))  # [256,64]
        x9 = self.conv3_7(x3 + x8)  # [256,64]
        x10 = self.conv2_3(self.upsample(x9))  # [128,128]
        x11 = self.conv3_8(x2 + x10)  # [128,128]
        x12 = self.conv2_4(self.upsample(x11))  # [64,256]
        x13 = self.conv3_9(x1 + x12)  # [64,256]
        x14 = self.conv3_10(x13)  # [1,256]
        return x14  # [1,256]
