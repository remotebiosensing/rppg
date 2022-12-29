import torch
import torch.nn as nn
from vid2bp.nets.modules.sub_modules.Trend_module import Trend_module_1D
from vid2bp.nets.modules.sub_modules.Detail_module import Detail_module_1D
from vid2bp.nets.modules.sub_modules.Linear_module import Linear_module_1D
from vid2bp.nets.modules.sub_modules.Amplitude_module import Amplitude_module
from torch.nn.functional import cosine_similarity
import matplotlib.pyplot as plt
from vid2bp.nets.modules.sub_modules.Frequency_block import frequency_block


class bvp2abp(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_val=4):
        super(bvp2abp, self).__init__()
        self.in_channel = in_channels
        self.out_channel = 32
        self.dilation_value = dilation_val
        # self.case = case
        # self.fft = fft
        self.trend_in_channel = 1
        self.detail_in_channel = in_channels - 1

        self.linear_model = Linear_module_1D()
        self.amplitude_model = Amplitude_module()
        self.trend_model = Trend_module_1D(self.trend_in_channel, self.out_channel)
        self.detail_model = Detail_module_1D(self.trend_in_channel, self.out_channel)
        # self.freq = frequency_block(self.trend_in_channel, self.out_channel)
        # self.channel_aggregation = nn.Conv1d(in_channels=self.out_channel * 2, out_channels=1, kernel_size=1, stride=1)
        # self.time_domain_fusion = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1, stride=1)
        # self.cross_domain_fusion = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1, stride=1)
        # self.cross_domain_fusion2 = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1, stride=1)
        # self.channel_fusion = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1, stride=1)

        # if self.case == 2:  # Trend + Detail
        #     self.trend_model = Trend_module_1D(self.trend_in_channel, self.out_channel)
        #     self.detail_model = Detail_module_1D(self.detail_in_channel, self.out_channel)
        # elif self.case == 1:  # Trend only
        #     self.trend_model = Trend_module_1D(self.trend_in_channel, self.out_channel)
        # else:  # Detail only
        #     self.detail_model = Detail_module_1D(self.detail_in_channel, self.out_channel)

    def forward(self, ple_input, scaler=True):
        # if self.in_channel==1:
        #     ple_sig = ple_input
        # else:
        batchN, channelN, seqN = ple_input.shape
        ple_sig = ple_input[:, 0, :].reshape(batchN, self.trend_in_channel, seqN)
        # dot = ple_sig @ ple_sig.transpose(1, 2)
        if scaler:
            norm = torch.norm(ple_sig, p=3, dim=2, keepdim=True)
            scaler = torch.max(norm) / norm
            ple_sig = ple_sig * scaler
        # div = torch.div(dot, norm)
        # mean = torch.mean(div)
        # scaler = torch.max(div) / norm
        # if channelN != 1:
        #     if self.detail_in_channel == 2:
        #         derivative_sig = ple_input[:, 1:2, :].reshape(batchN, self.detail_in_channel, seqN)
        #     else:
        #         derivative_sig = ple_input[:, 1:, :].reshape(batchN, self.detail_in_channel, seqN)
        # d, s, h = self.amplitude_model(ple_sig.view(batchN, seqN))

        d_out = self.detail_model(ple_sig)
        d_out = d_out.view(batchN, seqN)
        # t_out = self.trend_model(ple_sig).view(batchN, seqN)

        # s1, s2, s3, s4 = self.trend_model(ple_sig)
        # out = torch.mul((d - s), d_out) + h
        l_out = self.linear_model(d_out.view(batchN, seqN))

        return l_out#, d_out[0], d_fuse1[0], d_fuse2[0], d_fuse3[0]
