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
        self.enconv = torch.nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=32,
                      kernel_size=5, stride=1),
            nn.BatchNorm1d(32),
            nn.Conv1d(in_channels=32, out_channels=32,
                      kernel_size=5, stride=1),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout1d(0.5),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=32, out_channels=32,
                      kernel_size=5, stride=1),
            nn.BatchNorm1d(32),
            nn.Conv1d(in_channels=32, out_channels=32,
                      kernel_size=5, stride=1),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout1d(0.5),
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

    def forward(self, ple_input):
        at1 = self.enconv(ple_input)
        at2 = self.deconv(at1)

        return at1, at2
