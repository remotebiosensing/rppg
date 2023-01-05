import torch
import torch.nn as nn
from vid2bp.nets.modules.sub_modules.Trend_module import Trend_module_1D
from vid2bp.nets.modules.sub_modules.Detail_module import Detail_module_1D
from vid2bp.nets.modules.sub_modules.Linear_module import Linear_module_1D
from vid2bp.nets.modules.sub_modules.Amplitude_module import Amplitude_module
from vid2bp.nets.modules.sub_modules.residual_block import ResBlock_1D
from torch.nn.functional import cosine_similarity
import matplotlib.pyplot as plt
from vid2bp.nets.modules.sub_modules.Frequency_block import frequency_block
from vid2bp.nets.modules.ver2.FeatureExtractor import FeatureExtractor
from vid2bp.nets.modules.ver2.Amplifier import PleAmplifier


class bvp2abp(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_val=2):
        super(bvp2abp, self).__init__()
        self.in_channel = in_channels
        self.out_channels = out_channels
        self.dilation_value = dilation_val
        # self.case = case
        # self.fft = fft
        self.trend_in_channel = 1
        self.detail_in_channel = in_channels - 1

        self.linear_model = Linear_module_1D()
        self.amplitude_model = Amplitude_module()
        # self.channel_reduction = nn.Conv1d(in_channels=256, out_channels=1, kernel_size=1)
        self.feature_extractor = FeatureExtractor(in_channels=1, out_channels=self.out_channels)
        '''Amplify the small photoplethysmography signal'''
        self.ple_amplifier = PleAmplifier(in_channels=self.trend_in_channel)
        self.weights = nn.Parameter(torch.randn(360, 360))
        self.bias = nn.Parameter(torch.randn(256, 1))
        # self.res_module = ResBlock_1D(self.trend_in_channel, self.out_channel, dilation=dilation_val)
        # self.res_list = []
        # for i in range(4):
        #     in_ch = 1 if i == 0 else self.out_channel*2**i
        #     self.res_list.append(ResBlock_1D(in_channels=in_ch, out_channels=self.out_channel*2**i, kernel_size=3, dilation=dilation_val, idx=i+1).to('cuda:0'))

    def forward(self, ple_input, scaler=True):
        batchN, channelN, seqN = ple_input.shape
        ple_sig = ple_input[:, 0, :].reshape(batchN, self.trend_in_channel, seqN)
        # ple_sig = (ple_sig - torch.mean(ple_sig, dim=-1, keepdim=True)) / torch.std(ple_sig, dim=-1, keepdim=True)
        # amplification = ple_sig.view(256,-1) @ self.weights + self.bias
        scaled_ple = self.ple_amplifier(ple_sig)
        # l_out = self.linear_model(scaled_ple)
        f_out = self.feature_extractor(scaled_ple)
        l_out = self.linear_model(f_out.view(batchN, -1))
        ''''''

        return l_out, scaled_ple  # , d_out[0], d_fuse1[0], d_fuse2[0], d_fuse3[0]
