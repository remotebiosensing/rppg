import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import wandb

'''
nn.Conv1d expects as 3-dimensional input in the shape of [batch_size, channels, seq_len]
'''


class bvp2abp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(bvp2abp, self).__init__()
        self.in_channel = in_channels
        # 1D Convolution 3 size kernel (1@7500 -> 32@7500)
        self.conv_layer1 = torch.nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm1d(out_channels)
        )

        # Convolution 3x3 kernel Layer 2 (32@7500 -> 32@7500)
        self.conv_layer2 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                            kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

        self.conv_layer3 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=out_channels, out_channels=1,
                            kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm1d(1)
        )
        self.conv_layer4 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1,out_channels=1,kernel_size=1)
        )

        # Dropout
        self.a_dropout = torch.nn.Dropout1d(p=0.5)
        # Average-pooling 2x2 kernel Layer 3 (32@7500 -> 32@3750)
        self.avg_pool1 = torch.nn.AvgPool1d(kernel_size=2, stride=2)

        self.dense1 = torch.nn.Linear(900, 1000)
        self.dense2 = torch.nn.Linear(1000, 1800)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, ple_input):
        # print('shape of ple_input :', ple_input.shape)
        ple_input = torch.reshape(ple_input, (-1, self.in_channel, 1800))  # [ batch , channel, size]
        # print('reshape of ple_input :', ple_input.shape)
        c1 = self.conv_layer1(ple_input)
        # print('shape of conv1 output :', c1.shape)
        c2 = self.conv_layer2(c1)
        # print('shape of conv2 output :', c2.shape)
        d1 = self.a_dropout(c2)
        c3 = self.conv_layer3(d1)
        c4 = self.conv_layer4(c3)
        feature = self.sigmoid(c4)+c3
        # c3 = c3 + c2
        # print('shape of dropout output :', d1.shape)
        p1 = self.avg_pool1(feature)
        # print('shape of avg_pool1 output :', p1.shape)
        out = self.dense1(p1)
        out = self.dense2(out)

        # print('shape of dense output :', out.shape)
        return out
