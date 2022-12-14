import torch
import torch.nn as nn
from vid2bp.nets.modules.sub_modules.Trend_module import Trend_module_1D
from vid2bp.nets.modules.sub_modules.Detail_module import Detail_module_1D
from vid2bp.nets.modules.sub_modules.Linear_module import Linear_module_1D
from vid2bp.nets.modules.sub_modules.Amplitude_module import Amplitude_module
import matplotlib.pyplot as plt
from vid2bp.nets.modules.sub_modules.Frequency_block import frequency_block


class bvp2abp(nn.Module):
    def __init__(self, in_channels, out_channels, case, fft):
        super(bvp2abp, self).__init__()
        self.in_channel = in_channels
        self.out_channel = out_channels
        self.case = case
        self.fft = fft
        self.trend_in_channel = 1
        self.detail_in_channel = in_channels - 1

        self.linear_model = Linear_module_1D()
        self.amplitude_model = Amplitude_module()
        self.freq = frequency_block(self.trend_in_channel, self.out_channel)
        self.channel_aggregation = nn.Conv1d(in_channels=self.out_channel * 2, out_channels=1, kernel_size=1, stride=1)
        self.time_domain_fusion = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1, stride=1)
        self.cross_domain_fusion = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1, stride=1)
        self.cross_domain_fusion2 = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1, stride=1)
        self.channel_fusion = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1, stride=1)

        if self.case == 2:  # Trend + Detail
            self.trend_model = Trend_module_1D(self.trend_in_channel, self.out_channel)
            self.detail_model = Detail_module_1D(self.detail_in_channel, self.out_channel)
        elif self.case == 1:  # Trend only
            self.trend_model = Trend_module_1D(self.trend_in_channel, self.out_channel)
        else:  # Detail only
            self.detail_model = Detail_module_1D(self.detail_in_channel, self.out_channel)

    def forward(self, ple_input):
        # if self.in_channel==1:
        #     ple_sig = ple_input
        # else:
        batchN, channelN, seqN = ple_input.shape
        ple_sig = ple_input[:, 0, :].reshape(batchN, self.trend_in_channel, seqN)
        if self.in_channel != 1:
            if self.in_channel == 2:
                derivative_sig = ple_input[:, 1:2, :].reshape(batchN, self.detail_in_channel, seqN)
            else:
                derivative_sig = ple_input[:, 1:, :].reshape(batchN, self.detail_in_channel, seqN)

        # ppg signal만 들어올때
        if self.in_channel == 1:
            if self.case == 1:
                if self.fft == 1:  # PPG + Trend + FFT
                    t_out = self.channel_aggregation(self.trend_model.forward(ple_sig))
                    f_out = self.channel_aggregation(self.freq.forward(ple_sig))
                    # cross_domain_feature = self.cross_domain_fusion2(
                    #     torch.stack([torch.sum(t_out, dim=1), torch.sum(f_out, dim=1)], dim=1))
                    l_time = self.linear_model.forward(t_out)
                    l_freq = self.linear_model.forward(f_out)
                    l_out = self.cross_domain_fusion2(torch.stack([l_time, l_freq], dim=1))
                    # l_out = self.linear_model.forward(cross_domain_feature)
                else:  # PPG + Trend
                    t_out = self.channel_aggregation(self.trend_model.forward(ple_sig))
                    l_out = self.linear_model.forward(t_out)

        # ppg + vpg signal 들어올때
        elif self.in_channel == 2:
            if self.case == 2:
                if self.fft == 1:  # PPG + VPG + Trend + Detail + FFT
                    t_out = self.channel_aggregation(self.trend_model.forward(ple_sig))
                    d_out = self.channel_aggregation(self.detail_model.forward(derivative_sig))
                    f_out = self.channel_aggregation(self.freq.forward(ple_sig))
                    time_domain_feature = self.time_domain_fusion(torch.stack([torch.squeeze(t_out), torch.squeeze(d_out)], dim=1))
                    l_time = self.linear_model.forward(time_domain_feature)
                    l_freq = self.linear_model.forward(f_out)
                    l_out = self.cross_domain_fusion2(torch.stack([l_time, l_freq], dim=1))
                    # cross_domain_feature = self.cross_domain_fusion(
                    #     torch.stack([torch.sum(t_out, dim=1), torch.sum(d_out, dim=1), torch.sum(f_out, dim=1)], dim=1))
                    # l_out = self.linear_model.forward(cross_domain_feature)
                else:  # PPG + VPG + Trend + Detail
                    t_out = self.channel_aggregation(self.trend_model.forward(ple_sig))
                    d_out = self.channel_aggregation(self.detail_model.forward(derivative_sig))
                    time_domain_feature = self.time_domain_fusion(torch.stack([torch.squeeze(t_out), torch.squeeze(d_out)], dim=1))
                    # time_domain_feature = self.channel_fusion(
                    #     torch.stack([torch.sum(t_out, dim=1), torch.sum(d_out, dim=1)], dim=1))
                    l_out = self.linear_model.forward(time_domain_feature)
            elif self.case == 1:
                if self.fft == 1:  # PPG + VPG + Trend + FFT
                    t_out = self.channel_aggregation(self.trend_model.forward(ple_sig))
                    f_out = self.channel_aggregation(self.freq.forward(ple_sig))
                    l_time = self.linear_model.forward(t_out)
                    l_freq = self.linear_model.forward(f_out)
                    l_out = self.cross_domain_fusion2(torch.stack([l_time, l_freq], dim=1))
                    # cross_domain_feature = self.cross_domain_fusion2(
                    #     torch.stack([torch.sum(t_out, dim=1), torch.sum(f_out, dim=1)], dim=1))
                    # l_out = self.linear_model.forward(cross_domain_feature)
                else:  # PPG + VPG + Trend
                    t_out = self.trend_model.forward(ple_sig)
                    l_out = self.linear_model.forward(torch.sum(t_out, dim=1).unsqueeze(1))
            else:
                if self.fft == 1:  # PPG + VPG + Detail + FFT
                    d_out = self.channel_aggregation(self.detail_model.forward(derivative_sig))
                    f_out = self.channel_aggregation(self.freq.forward(ple_sig))
                    l_time = self.linear_model.forward(d_out)
                    l_freq = self.linear_model.forward(f_out)
                    l_out = self.cross_domain_fusion2(torch.stack([l_time, l_freq], dim=1))
                    # cross_domain_feature = self.cross_domain_fusion2(
                    #     torch.stack([torch.sum(d_out, dim=1), torch.sum(f_out, dim=1)], dim=1))
                    # l_out = self.linear_model.forward(cross_domain_feature)
                else:  # PPG + VPG + Detail
                    d_out = self.channel_aggregation(self.detail_model.forward(derivative_sig))
                    l_out = self.linear_model.forward(d_out)

        # ppg + vpg + apg signal 들어올때
        else:
            if self.case == 2:
                if self.fft == 1:  # PPG + VPG + APG + Trend + Detail + FFT
                    t_out = self.channel_aggregation(self.trend_model.forward(ple_sig))
                    d_out = self.channel_aggregation(self.detail_model.forward(derivative_sig))
                    f_out = self.freq.forward(ple_sig)
                    time_domain_feature = self.time_domain_fusion(torch.stack([torch.squeeze(t_out), torch.squeeze(d_out)], dim=1))
                    l_time = self.linear_model.forward(time_domain_feature)
                    l_freq = self.linear_model.forward(f_out.unsqueeze(1))
                    l_out = self.cross_domain_fusion2(torch.stack([l_time, l_freq], dim=1))
                    # cross_domain_feature = self.cross_domain_fusion(
                    #     torch.stack([torch.sum(t_out, dim=1), torch.sum(d_out, dim=1), torch.sum(f_out, dim=1)], dim=1))
                    # l_out = self.linear_model.forward(cross_domain_feature)
                else:  # PPG + VPG + APG + Trend + Detail
                    t_out = self.channel_aggregation(self.trend_model.forward(ple_sig))
                    d_out = self.channel_aggregation(self.detail_model.forward(derivative_sig))
                    time_domain_feature = self.time_domain_fusion(torch.stack([torch.squeeze(t_out), torch.squeeze(d_out)], dim=1))
                    # time_domain_feature = self.channel_fusion(
                    #     torch.stack([torch.sum(t_out, dim=1), torch.sum(d_out, dim=1)], dim=1))
                    l_out = self.linear_model.forward(time_domain_feature)
            elif self.case == 1:
                if self.fft == 1:  # PPG + VPG + APG + Trend + FFT
                    t_out = self.channel_aggregation(self.trend_model.forward(ple_sig))
                    f_out = self.channel_aggregation(self.freq.forward(ple_sig))
                    l_time = self.linear_model.forward(t_out)
                    l_freq = self.linear_model.forward(f_out)
                    l_out = self.cross_domain_fusion2(torch.stack([l_time, l_freq], dim=1))
                    # cross_domain_feature = self.cross_domain_fusion2(
                    #     torch.stack([torch.sum(t_out, dim=1), torch.sum(f_out, dim=1)], dim=1))
                    # l_out = self.linear_model.forward(cross_domain_feature)
                else:  # PPG + VPG + APG + Trend
                    t_out = self.channel_aggregation(self.trend_model.forward(ple_sig))
                    l_out = self.linear_model.forward(t_out)
            else:
                if self.fft == 1:  # PPG + VPG + APG + Detail + FFT
                    d_out = self.channel_aggregation(self.detail_model.forward(derivative_sig))
                    f_out = self.channel_aggregation(self.freq.forward(ple_sig))
                    l_time = self.linear_model.forward(d_out)
                    l_freq = self.linear_model.forward(f_out)
                    l_out = self.cross_domain_fusion2(torch.stack([l_time, l_freq], dim=1))
                    #
                    # cross_domain_feature = self.cross_domain_fusion2(
                    #     torch.stack([torch.sum(d_out, dim=1), torch.sum(f_out, dim=1)], dim=1))
                    # l_out = self.linear_model.forward(cross_domain_feature)
                else:  # PPG + VPG + APG + Detail
                    d_out = self.channel_aggregation(self.detail_model.forward(derivative_sig))
                    l_out = self.linear_model.forward(d_out)

        # if self.case == 2:
        #     if self.fft==1:
        #         t_out = self.trend_model.forward(ple_sig)  # shape: [4096, 32, 84] [4096, 1, 100]
        #         d_out = self.detail_model.forward(derivative_sig)  # shape: [4096, 1, 100]
        #         f_out = self.freq.forward(ple_sig)
        #         cross_domain_feature = self.cross_domain_fusion(torch.stack([torch.sum(t_out, dim=1), torch.sum(d_out, dim=1), torch.sum(f_out, dim=1)], dim=1))
        #         l_out = self.linear_model.forward(cross_domain_feature)
        #     else:
        #         t_out = self.trend_model.forward(ple_sig)
        #         d_out = self.detail_model.forward(derivative_sig)
        #         time_domain_feature = self.channel_fusion(torch.stack([torch.sum(t_out, dim=1), torch.sum(d_out, dim=1)], dim=1))
        #         l_out = self.linear_model.forward(time_domain_feature)
        #
        # elif self.case == 1:
        #     t_out = self.trend_model.forward(ple_sig)
        #     f_out = self.freq.forward(ple_sig)
        #     cross_domain_feature = self.cross_domain_fusion(torch.stack([torch.sum(t_out, dim=1), torch.sum(f_out, dim=1)], dim=1))
        #     l_out = self.linear_model.forward(cross_domain_feature)
        # else:
        #     d_out = self.detail_model.forward(derivative_sig)
        #     f_out = self.freq.forward(ple_sig)
        #     cross_domain_feature = self.cross_domain_fusion(torch.stack([torch.sum(d_out, dim=1), torch.sum(f_out, dim=1)], dim=1))
        #     l_out = self.linear_model.forward(cross_domain_feature)

        return l_out  # , dbp, sbp
