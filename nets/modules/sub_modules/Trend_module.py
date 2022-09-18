import torch
import torch.nn as nn

import json

with open('/home/paperc/PycharmProjects/VBPNet/config/parameter.json') as f:
    json_data = json.load(f)
    param = json_data.get("parameters")
    in_channel = json_data.get("parameters").get("in_channels")
    sampling_rate = json_data.get("parameters").get("sampling_rate")

'''
nn.Conv1d expects as 3-dimensional input in the shape of [batch_size, channels, seq_len]
'''

class Trend_module_1D(nn.Module):
    def __init__(self, in_channels):
        super(Trend_module_1D, self).__init__()
        self.in_channel = in_channels
        # 1D Convolution 3 size kernel (1@7500 -> 32@7500)
        self.enconv = torch.nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=32,
                      kernel_size=5, stride=1),
            nn.BatchNorm1d(32),
            nn.Conv1d(in_channels=32, out_channels=32,
                      kernel_size=5, stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=32, out_channels=32,
                      kernel_size=5, stride=1),
            nn.BatchNorm1d(32),
            nn.Conv1d(in_channels=32, out_channels=32,
                      kernel_size=5, stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.deconv = torch.nn.Sequential(
            nn.ConvTranspose1d(in_channels=32, out_channels=32,
                               kernel_size=5, stride=1),
            nn.BatchNorm1d(32),
            nn.ConvTranspose1d(in_channels=32, out_channels=32,
                               kernel_size=5, stride=1),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.ConvTranspose1d(in_channels=32, out_channels=32,
                               kernel_size=5, stride=1),
            nn.BatchNorm1d(32),
            nn.ConvTranspose1d(in_channels=32, out_channels=1,
                               kernel_size=5, stride=1),
            nn.BatchNorm1d(1),
            nn.ELU()
        )

    # TODO FORWARD안에 FEATURE 앞에다가 DATALOADER(__GETITEM__())에서 얻은 크기 정보 추가
    def forward(self, ple_input, at1, at2):
        # ple_input = torch.reshape(ple_input, (-1, self.in_channel, 360))  # [ batch , channel, size]
        out = self.enconv(ple_input)
        out = at1 + out
        out = self.deconv(out)
        out = at2 + out

        return out
