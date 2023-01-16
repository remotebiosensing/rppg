import torch
import torch.nn as nn
from vid2bp.nets.modules.sub_modules.Trend_module import Trend_module_1D
from vid2bp.nets.modules.sub_modules.Detail_module import Detail_module_1D
from vid2bp.nets.modules.sub_modules.Linear_module import Linear_module_1D, Linear_module_1D_dy, Linear_module_1D_ddy
from vid2bp.nets.modules.sub_modules.Amplitude_module import Amplitude_module
from vid2bp.nets.modules.sub_modules.residual_block import ResBlock_1D
from torch.nn.functional import cosine_similarity
import matplotlib.pyplot as plt
from vid2bp.nets.modules.sub_modules.Frequency_block import frequency_block
from vid2bp.nets.modules.ver2.FeatureExtractor import FeatureExtractor, FeatureExtractor1, FeatureExtractor2
from vid2bp.nets.modules.ver2.Amplifier import PleAmplifier
from vid2bp.nets.modules.ver2.PositionalEncoding import PositionalEncoding

class bvp2abp(nn.Module):
    def __init__(self, in_channels, out_channels, target_samp_rate=60, dilation_val=2):
        super(bvp2abp, self).__init__()
        self.in_channel = in_channels
        self.out_channels = out_channels
        self.dilation_value = dilation_val
        # self.case = case
        # self.fft = fft
        self.trend_in_channel = 1
        self.detail_in_channel = in_channels - 1
        self.p_encoder = PositionalEncoding(360, 1024)
        self.p_encoder2 = PositionalEncoding(180, 1024)
        self.channel_pool = nn.Conv1d(out_channels, 1, kernel_size=1)

        self.linear_model = Linear_module_1D()
        self.linear_model1 = Linear_module_1D_dy()
        self.linear_model2 = Linear_module_1D_ddy()
        self.amplitude_model = Amplitude_module()
        # self.channel_reduction = nn.Conv1d(in_channels=256, out_channels=1, kernel_size=1)
        self.feature_extractor = FeatureExtractor(in_channels=2, out_channels=self.out_channels, dilation_val=self.dilation_value)
        self.feature_extractor1 = FeatureExtractor1(in_channels=2, out_channels=self.out_channels, dilation_val=self.dilation_value)
        self.feature_extractor2 = FeatureExtractor2(in_channels=2, out_channels=self.out_channels, dilation_val=self.dilation_value)
        '''Amplify the small photoplethysmography signal'''
        self.ple_amplifier = PleAmplifier(in_channels=self.trend_in_channel)
        # self.weights = nn.Parameter(torch.randn(360, 360))
        # self.bias = nn.Parameter(torch.randn(256, 1))
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, dilation=2)
        # self.res_module = ResBlock_1D(self.trend_in_channel, self.out_channel, dilation=dilation_val)
        # self.res_list = []
        # for i in range(4):
        #     in_ch = 1 if i == 0 else self.out_channel*2**i
        #     self.res_list.append(ResBlock_1D(in_channels=in_ch, out_channels=self.out_channel*2**i, kernel_size=3, dilation=dilation_val, idx=i+1).to('cuda:0'))

    def forward(self, x, dx, ddx, scaler=True):
        batchN, seqN = x.shape
        # x = x.unsqueeze(1)
        # test = self.conv1(x)
        scaled_ple = self.ple_amplifier(x)
        px = self.p_encoder(scaled_ple)
        pdx = self.p_encoder2(dx)
        pddx = self.p_encoder2(ddx)
        f_out = self.channel_pool(self.feature_extractor(px))
        f_out1 = self.channel_pool(self.feature_extractor1(pdx))
        f_out2 = self.channel_pool(self.feature_extractor2(pddx))
        y = self.linear_model(f_out).view(batchN, seqN)
        dy = self.linear_model1(f_out1).view(batchN, seqN//2)
        ddy = self.linear_model2(f_out2).view(batchN, seqN//2)

        # ple_sig = x[:, 0, :].reshape(batchN, self.trend_in_channel, seqN)
        # ple_sig = (ple_sig - torch.mean(ple_sig, dim=-1, keepdim=True)) / torch.std(ple_sig, dim=-1, keepdim=True)
        # amplification = ple_sig.view(256,-1) @ self.weights + self.bias
        # f_out = self.feature_extractor(scaled_ple)
        # l_out = self.linear_model(f_out.view(batchN, -1))
        ''''''

        return y, dy, ddy, scaled_ple  # , d_out[0], d_fuse1[0], d_fuse2[0], d_fuse3[0]
