import torch
import torch.nn as nn
from torchsummary import summary


def convert_1d_to_2d(x, width=25):
    ppg_signal = x[:, 0:1, :]
    ppg_2d = ppg_signal.view(ppg_signal.shape[0], 1, width, -1)
    # x = x.view(batchN, channelN, seqN, 1)
    return ppg_2d


def conv2d_3(ch_out=4, kernel=3):
    return nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=ch_out, kernel_size=kernel, stride=1),
        nn.BatchNorm2d(ch_out),
        nn.Conv2d(in_channels=ch_out, out_channels=1, kernel_size=kernel, stride=1),
        nn.BatchNorm2d(1),
        nn.ReLU6(),
        nn.MaxPool2d(kernel_size=2)
    )


def linear2d(in_feat=10):
    return nn.Sequential(
        nn.Linear(in_features=in_feat, out_features=10),
        nn.Linear(in_features=10, out_features=2),
        nn.Sigmoid()
    )


class Film1Dto2D(nn.Module):
    def __init__(self, width=25):
        super(Film1Dto2D, self).__init__()
        self.width = width
        self.conv = conv2d_3(ch_out=4, kernel=3)
        self.linear = linear2d(130)

    def forward(self, x):
        converted_2d = convert_1d_to_2d(x, self.width)
        c1 = self.conv(converted_2d)
        c1 = c1.view(c1.shape[0], -1)
        l1 = self.linear(c1)
        return l1


if __name__ == "__main__":
    model = Film1Dto2D(width=25)
    summary(model, (3, 750), batch_size=1024, device='cpu')
