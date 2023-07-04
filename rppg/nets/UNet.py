import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, input_channels):
        super(UNet, self).__init__()

        # assert input_channels
        self.input_channels = input_channels

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2,
                                    mode='bilinear',
                                    align_corners=True)

        # Encode
        self.double_conv_1 = double_conv_bn_relu(input_channels, 64)
        self.double_conv_2 = double_conv_bn_relu(64, 128)
        self.double_conv_3 = double_conv_bn_relu(128, 256)
        self.double_conv_4 = double_conv_bn_relu(256, 256)

        # Decode
        self.deconv2d_1 = double_conv_bn_relu(512, 128)
        self.deconv2d_2 = double_conv_bn_relu(256, 64)
        self.deconv2d_3 = double_conv_bn_relu(128, 64)

        self.final_double_conv = double_conv_bn_relu(64, 64)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encode
        x1 = self.double_conv_1(x)
        x2 = self.max_pool(x1)
        x3 = self.double_conv_2(x2)
        x4 = self.max_pool(x3)
        x5 = self.double_conv_3(x4)
        x6 = self.max_pool(x5)
        x7 = self.double_conv_4(x6)

        # Decode
        x8 = self.upsample(x7)
        x8 = torch.cat([x8, x5], dim=1)
        x8 = self.deconv2d_1(x8)

        x9 = self.upsample(x8)
        x9 = torch.cat([x9, x3], dim=1)
        x9 = self.deconv2d_2(x9)

        x10 = self.upsample(x9)
        x10 = torch.cat([x10, x1], dim=1)
        x10 = self.deconv2d_3(x10)

        x11 = self.final_double_conv(x10)
        predictions = self.sigmoid(self.final_conv(x11))
        return predictions


def double_conv_bn_relu(in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
    conv1 = nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=kernel_size,
                      padding=padding)
    leaky_relu_1 = nn.LeakyReLU(inplace=True)
    bn_1 = nn.BatchNorm2d(out_channels)
    conv2 = nn.Conv2d(out_channels,
                      out_channels,
                      kernel_size=kernel_size,
                      padding=padding)
    leaky_relu_2 = nn.LeakyReLU(inplace=True)
    bn_2 = nn.BatchNorm2d(out_channels)
    return nn.Sequential(conv1, leaky_relu_1, bn_1, conv2, leaky_relu_2, bn_2)
