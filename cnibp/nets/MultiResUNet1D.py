import torch
import torch.nn as nn


class conv2d_bn(nn.Module):
    def __init__(self, in_channel, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu',
                 name=None):
        super(conv2d_bn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=filters, kernel_size=3, padding=padding),
            nn.BatchNorm1d(num_features=filters)
        )
        if activation is None:
            pass
        else:
            self.conv = nn.Sequential(self.conv, nn.ReLU())

    def forward(self, x):
        x = self.conv(x)
        return x


class MultiResBlock(nn.Module):
    def __init__(self, U, in_channel, alpha=2.5):
        super(MultiResBlock, self).__init__()
        self.W = alpha * U
        self.shortcut = conv2d_bn(in_channel, filters=int(self.W * 0.167) + int(self.W * 0.333) + int(self.W * 0.5),
                                  num_col=1, num_row=1, activation=None, padding='same')
        self.conv3x3 = conv2d_bn(in_channel, filters=int(self.W * 0.167), num_col=3, num_row=3,
                                 activation='relu', padding='same')
        self.conv5x5 = conv2d_bn(int(self.W * 0.167), filters=int(self.W * 0.333), num_col=3, num_row=3,
                                 activation='relu', padding='same')
        self.conv7x7 = conv2d_bn(int(self.W * 0.333), filters=int(self.W * 0.5), num_col=3, num_row=3,
                                 activation='relu', padding='same')
        self.batchnorm = nn.BatchNorm1d(num_features=int(self.W * 0.167) + int(self.W * 0.333) + int(self.W * 0.5))

    def forward(self, x):
        shortcut = self.shortcut(x)
        conv3x3 = self.conv3x3(x)
        conv5x5 = self.conv5x5(conv3x3)
        conv7x7 = self.conv7x7(conv5x5)
        x = torch.cat([conv3x3, conv5x5, conv7x7], dim=1)
        x = self.batchnorm(x)
        x += shortcut
        x = nn.ReLU()(x)
        x = self.batchnorm(x)

        return x


class ResPath(nn.Module):
    def __init__(self, in_channel, filters, length):
        super(ResPath, self).__init__()
        self.in_channel = in_channel
        self.length = length
        self.shortcut = conv2d_bn(in_channel, filters, 1, 1, activation=None, padding='same')
        self.conv = conv2d_bn(in_channel, filters, 3, 3, activation='relu', padding='same')
        self.ReLu = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(num_features=filters)
        if length > 1:
            self.respath = ResPath(filters, filters, length - 1)

    def forward(self, x):
        shortcut = self.shortcut(x)
        conv = self.conv(x)
        x = shortcut + conv
        x = self.ReLu(x)
        x = self.batchnorm(x)
        for i in range(self.length - 1):
            x = self.respath(x)

        return x


class MultiResUNet1D(nn.Module):
    def __init__(self):
        super(MultiResUNet1D, self).__init__()
        self.mresblock1 = MultiResBlock(U=32, in_channel=1)  # out_channel = 79
        self.pool = nn.MaxPool1d(2)
        self.ResPath1 = ResPath(filters=32, length=4, in_channel=79)  # out_channel = 32

        self.mresblock2 = MultiResBlock(U=64, in_channel=79)  # out_channel = 159
        self.ResPath2 = ResPath(in_channel=159, filters=64, length=3)  # out_channel = 64

        self.mresblock3 = MultiResBlock(U=128, in_channel=159)
        # out_channel = 319
        self.ResPath3 = ResPath(in_channel=319, filters=128, length=2)  # out_channel = 128

        self.mresblock4 = MultiResBlock(U=256, in_channel=319)  # out_channel = 639
        self.ResPath4 = ResPath(in_channel=639, filters=256, length=1)  # out_channel = 256

        self.mresblock5 = MultiResBlock(U=512, in_channel=639)  # out_channel = 1279

        self.up6 = nn.Upsample(scale_factor=2)  # out_channel = 1279
        # cat up6, ResPath4 -> out_channel = 1279 + 256 = 1535
        self.mresblock6 = MultiResBlock(U=256, in_channel=1535)  # out_channel = 639

        self.up7 = nn.Upsample(scale_factor=2)  # out_channel = 639
        # cat up7, ResPath3 -> out_channel = 639 + 128 = 767
        self.mresblock7 = MultiResBlock(U=128, in_channel=767)  # out_channel = 319

        self.up8 = nn.Upsample(scale_factor=2)  # out_channel = 318
        # cat up8, ResPath2 -> out_channel = 318 + 64 = 383
        self.mresblock8 = MultiResBlock(U=64, in_channel=383)  # out_channel = 159

        self.up9 = nn.Upsample(scale_factor=2)  # out_channel = 159
        # cat up9, ResPath1 -> out_channel = 159 + 32 = 191
        self.mresblock9 = MultiResBlock(U=32, in_channel=191)  # out_channel = 79

        self.conv10 = nn.Conv1d(79, 1, 1, 1, padding='same')

    def forward(self, x):
        mresblock1 = self.mresblock1(x)
        pool1 = self.pool(mresblock1)
        mresblock1 = self.ResPath1(mresblock1)

        mresblock2 = self.mresblock2(pool1)
        pool2 = self.pool(mresblock2)
        mresblock2 = self.ResPath2(mresblock2)

        mresblock3 = self.mresblock3(pool2)
        pool3 = self.pool(mresblock3)
        mresblock3 = self.ResPath3(mresblock3)

        mresblock4 = self.mresblock4(pool3)
        pool4 = self.pool(mresblock4)
        mresblock4 = self.ResPath4(mresblock4)

        mresblock5 = self.mresblock5(pool4)

        up6 = self.up6(mresblock5)
        mresblock6 = self.mresblock6(torch.cat([up6, mresblock4], dim=1))

        up7 = self.up7(mresblock6)
        mresblock7 = self.mresblock7(torch.cat([up7, mresblock3], dim=1))

        up8 = self.up8(mresblock7)
        mresblock8 = self.mresblock8(torch.cat([up8, mresblock2], dim=1))

        up9 = self.up9(mresblock8)
        mresblock9 = self.mresblock9(torch.cat([up9, mresblock1], dim=1))

        conv10 = self.conv10(mresblock9)

        return conv10
