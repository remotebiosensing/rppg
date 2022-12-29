import torch
import torch.nn as nn
from vid2bp.nets.modules.sub_modules.Attention_module import Attention_module_1D
import matplotlib.pyplot as plt

import json

# with open('/home/paperc/PycharmProjects/Pytorch_rppgs/vid2bp/config/parameter.json') as f:
#     json_data = json.load(f)
#     param = json_data.get("parameters")
#     in_channel = json_data.get("parameters").get("in_channels")
#     sampling_rate = json_data.get("parameters").get("sampling_rate")

'''
nn.Conv1d expects as 3-dimensional input in the shape of [batch_size, channels, seq_len]
'''

''' 원래 값의 normalized attention을 return하는 모듈 '''
''' 왜 normalized attention을 구하냐면, 사람마다 ppg gain이 다르기 때문이다.'''

class Trend_module_1D(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_value=4):
        super(Trend_module_1D, self).__init__()
        # self.attention1 = Attention_module_1D(out_channels)
        # self.attention2 = Attention_module_1D(out_channels * 2)
        # self.freq = frequency_block()
        self.dropout = nn.Dropout1d(p=0.5)
        self.max_pool = nn.MaxPool1d(2)
        self.avg_pool = nn.AvgPool1d(2)
        self.elu = nn.ELU()
        self.upsampler2 = nn.Upsample(scale_factor=2, mode='linear')
        self.upsampler4 = nn.Upsample(scale_factor=4, mode='linear')
        self.upsampler8 = nn.Upsample(scale_factor=8, mode='linear')
        self.enconv1 = nn.Sequential(  # [batch, in_channels, 360] -> [batch, out_channels, 360]
            # nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, dilation=dilation_value, padding=dilation_value),
            nn.BatchNorm1d(out_channels)
        )
        self.enconv2 = nn.Sequential(  # [batch, out_channels, 360] -> [batch, out_channels, 360]
            # nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, dilation=dilation_value, padding=dilation_value),
            nn.BatchNorm1d(out_channels),
            nn.AvgPool1d(2)
        )
        self.enconv3 = nn.Sequential(  # [batch, out_channel, 180] -> [batch, out_channel*2, 180]
            # nn.Conv1d(out_channels, out_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, dilation=dilation_value, padding=dilation_value),
            nn.BatchNorm1d(out_channels),
            nn.AvgPool1d(2)
        )
        self.enconv4 = nn.Sequential(  # [batch, out_channels*2, 180] -> [batch, out_channels*2, 180]
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, dilation=dilation_value, padding=dilation_value),
            nn.BatchNorm1d(out_channels),
            nn.AvgPool1d(2)
        )
        self.channel_attention = nn.Sequential(
            nn.Conv1d(out_channels, 1, kernel_size=1, stride=1),
            nn.BatchNorm1d(1),
            nn.Dropout1d(0.5)
        )
        self.relu = nn.ReLU()


    def forward(self, ple):  # [batch, 1, 360]
        # f1 = self.enconv1(self.freq(ple).unsqueeze(1))
        # t1 = self.enconv1(ple)
        # t2 = self.enconv2(t1)
        # t_fuse1 = (t1 + self.upsampler2(t2)) * self.upsampler2(t2)
        # t3 = self.enconv3(t2)
        # t_fuse2 = (t_fuse1 + self.upsampler4(t3)) * self.upsampler4(t3)
        # t4 = self.enconv4(t3)
        # t_fuse3 = (t_fuse2 + self.upsampler8(t4)) * self.upsampler8(t4)
        # return t_fuse1, t_fuse2, t_fuse3

        # t1 = self.enconv1(ple)
        # s1 = t1 + ple
        # t2 = self.enconv2(self.dropout(self.elu(s1)))
        # s2 = s1 + self.upsampler2(t2)
        # t3 = self.enconv3(self.dropout(self.elu(s2)))
        # s3 = s2 + self.upsampler2(t3)
        # t4 = self.enconv4(self.dropout(self.elu(s3)))
        # s4 = t3 + self.upsampler2(t4)
        # return s1, s2, s3, s4


        # t1 = self.enconv1(ple)  # [batch, 16, 360]
        # f1 = (ple+t1) * t1
        # t2 = self.enconv2(t1)  # [batch, 16, 360]
        # f2 = (f1 + t2) * t2
        # # at1 = self.attention1(t2)  # [batch, 1, 360]
        # p1 = self.max_pool(self.dropout(f2))  # [batch, 1, 180]
        # # f2 = self.enconv3(self.freq(torch.t3))
        # t3 = self.enconv3(p1)  # [batch, 32, 180]
        # f3 = (p1 + t3) * t3
        # t4 = self.enconv4(t3)  # [batch, 32, 180]
        # f4 = (f3 + t4) * t4
        # # at2 = self.attention2(t4)  # [batch, 1, 180]
        # p2 = self.max_pool(self.dropout(f4))  # [batch, 32, 90]
        # # test = torch.mean(t4, dim=1) # [batch, 180]
        # return t1, t2, t3, t4 #at1, at2  # [batch, 32, 90]

        t1 = self.enconv1(ple)  # [batch, 16, 360]
        t2 = self.enconv2(t1)  # [batch, 16, 360]
        t_fuse1 = (t1 + self.upsampler2(t2)) * self.upsampler2(t2)
        t3 = self.enconv3(t2)  # [batch, 32, 180]
        t_fuse2 = (t_fuse1 + self.upsampler4(t3)) * self.upsampler4(t3)
        t4 = self.enconv4(t3)  # [batch, 32, 180]
        t_fuse3 = (t_fuse2 + self.upsampler8(t4)) * self.upsampler8(t4)
        out = self.channel_attention(t_fuse3)

        return out

