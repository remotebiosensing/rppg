import torch
import torch.nn as nn
from vid2bp.nets.modules.sub_modules.Attention_module import Attention_module_1D
import matplotlib.pyplot as plt

import json

with open('/home/paperc/PycharmProjects/Pytorch_rppgs/vid2bp/config/parameter.json') as f:
    json_data = json.load(f)
    param = json_data.get("parameters")
    in_channel = json_data.get("parameters").get("in_channels")
    sampling_rate = json_data.get("parameters").get("sampling_rate")

'''
nn.Conv1d expects as 3-dimensional input in the shape of [batch_size, channels, seq_len]
'''

''' 원래 값의 normalized attention을 return하는 모듈 '''
''' 왜 normalized attention을 구하냐면, 사람마다 ppg gain이 다르기 때문이다.'''


class Trend_module_1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Trend_module_1D, self).__init__()
        self.attention1 = Attention_module_1D(out_channels)
        self.attention2 = Attention_module_1D(out_channels * 2)
        # self.freq = frequency_block()

        self.enconv1 = nn.Sequential(  # [batch, in_channels, 360] -> [batch, out_channels, 360]
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm1d(out_channels)
        )
        self.enconv2 = nn.Sequential(  # [batch, out_channels, 360] -> [batch, out_channels, 360]
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm1d(out_channels),
            nn.ELU()
        )
        self.enconv3 = nn.Sequential(  # [batch, out_channel, 180] -> [batch, out_channel*2, 180]
            nn.Conv1d(out_channels, out_channels * 2, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm1d(out_channels * 2)
        )
        self.enconv4 = nn.Sequential(  # [batch, out_channels*2, 180] -> [batch, out_channels*2, 180]
            nn.Conv1d(out_channels * 2, out_channels * 2, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm1d(out_channels * 2),
            nn.ELU()
        )
        self.dropout = nn.Dropout1d(p=0.5)
        self.max_pool = nn.MaxPool1d(2)
        self.elu = nn.ELU()

    def forward(self, ple):  # [batch, 1, 360]
        # f1 = self.enconv1(self.freq(ple).unsqueeze(1))
        t1 = self.enconv1(ple)  # [batch, 16, 360]
        t2 = self.enconv2(t1)  # [batch, 16, 360]
        # at1 = self.attention1(t2)  # [batch, 1, 360]

        p1 = self.max_pool(self.dropout(t2))  # [batch, 1, 180]
        # f2 = self.enconv3(self.freq(torch.t3))
        t3 = self.enconv3(p1)  # [batch, 32, 180]
        t4 = self.enconv4(t3)  # [batch, 32, 180]
        # at2 = self.attention2(t4)  # [batch, 1, 180]
        p2 = self.max_pool(self.dropout(t4))  # [batch, 32, 90]
        # test = torch.mean(t4, dim=1) # [batch, 180]
        return p2  # [batch, 32, 90]
