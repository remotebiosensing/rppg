import torch
import torch.nn as nn


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.maxpool = nn.MaxPool1d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.LeakyReLU())  # first conv layer output_shape
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1),
            nn.LeakyReLU(), )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1),
            nn.LeakyReLU(), )
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Dropout1d(0.5))
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Dropout1d(0.5))
        self.conv6 = nn.Sequential(
            nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=2, stride=1),
            nn.LeakyReLU())
        self.conv7 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1),
            nn.LeakyReLU())
        self.conv8 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=2, stride=1),
            nn.LeakyReLU())
        self.conv9 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1),
            nn.LeakyReLU())
        self.conv10 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=2, stride=1),
            nn.LeakyReLU())
        self.conv11 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1),
            nn.LeakyReLU())
        self.conv12 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=2, stride=1),
            nn.LeakyReLU())
        self.conv13 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.LeakyReLU())
        self.conv14 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=2, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=2, out_channels=1, kernel_size=3, stride=1),
            nn.LeakyReLU())

    def forward(self, x):
        x1 = self.conv1(x)  # output_shape  (64, 256)
        x2 = self.conv2(self.maxpool(x1))  # output_shape  (128, 128)
        x3 = self.conv3(self.maxpool(x2))  # output_shape  (256, 64)
        x4 = self.conv4(self.maxpool(x3))  # output_shape  (512, 32)
        x5 = self.conv5(self.maxpool(x4))  # output_shape  (1024, 16)
        x6 = self.conv6(self.upsample(x5))  # output_shape  (512, 32)
        x7 = self.conv7(torch.cat([x6, x4], dim=1))  # output_shape  (512, 32)
        x8 = self.conv8(self.upsample(x7))  # output_shape  (256, 64)
        x9 = self.conv9(torch.cat([x8, x3], dim=1))  # output_shape  (256, 64)
        x10 = self.conv10(self.upsample(x9))  # output_shape  (128, 128)
        x11 = self.conv11(torch.cat([x10, x2], dim=1))  # output_shape  (128, 128)
        x12 = self.conv12(self.upsample(x11))  # output_shape  (64, 256)
        x13 = self.conv13(torch.cat([x12, x1], dim=1))  # output_shape  (64, 256)
        x14 = self.conv14(x13)  # output_shape  (1, 256)
        return x14  # output_shape  (1, 256)
