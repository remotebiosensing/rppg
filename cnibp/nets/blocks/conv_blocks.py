import torch
import torch.nn as nn
from torchsummary import summary


def dwise_conv1d(ch_in, ker=3, stride=1):
    pad = ker // 2
    return (
        nn.Sequential(
            nn.Conv1d(ch_in, ch_in, kernel_size=ker, padding=pad, stride=stride, groups=ch_in, bias=False),
            nn.BatchNorm1d(ch_in),
            # nn.Conv1d(ch_in, ch_in, kernel_size=ker, padding=pad, stride=stride, groups=ch_in, bias=False),
            # nn.BatchNorm1d(ch_in),
            nn.ReLU6(inplace=True)
        )
    )


def conv1d_1x1(ch_in, ch_out):
    if ch_in < ch_out:
        return (
            nn.Sequential(
                nn.Conv1d(ch_in, ch_out, kernel_size=1, padding=0, stride=1, bias=False),
                nn.BatchNorm1d(ch_out),
                nn.ReLU6(inplace=True)
            )
        )
    else:
        return (
            nn.Sequential(
                nn.Conv1d(ch_in, ch_out, kernel_size=1, padding=0, stride=1, bias=False),
                nn.BatchNorm1d(ch_out)
            )
        )


def conv1d_3(ch_in, ch_out, stride):
    return (
        nn.Sequential(
            nn.Conv1d(ch_in, ch_out, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm1d(ch_out),
            nn.ReLU6(inplace=True)
        )
    )


class InvertedBlock(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, expand_ratio, stride):
        super(InvertedBlock, self).__init__()

        self.stride = stride
        self.use_res_connect = self.stride == 1 and ch_in == ch_out

        layers = []

        hidden_dims = ch_in * expand_ratio
        if expand_ratio != 1:
            # point-wise for channel expansion corresponding to the prior Block
            layers.append(conv1d_1x1(ch_in, hidden_dims))
        layers.extend([
            # depth-wise
            dwise_conv1d(hidden_dims, kernel_size, stride=self.stride),
            # point-wise
            conv1d_1x1(hidden_dims, ch_out)
        ])

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.layers(x)
        else:
            return self.layers(x)

