import torch
import torch.nn as nn
# from vid2bp.nets.modules.sub_modules.Trend_module import Trend_module_1D
# from vid2bp.nets.modules.sub_modules.Detail_module import Detail_module_1D
from vid2bp.nets.modules.sub_modules.Linear_module import Linear_module_1D
from vid2bp.nets.modules.sub_modules.Amplitude_module import DBP_module, SBP_module
# from vid2bp.nets.modules.sub_modules.residual_block import ResBlock_1D
# from torch.nn.functional import cosine_similarity
import matplotlib.pyplot as plt
# from vid2bp.nets.modules.sub_modules.Frequency_block import frequency_block
from vid2bp.nets.modules.ver2.Detail_module import DetailFeatureExtractor2D
from vid2bp.nets.modules.ver2.Trend_module import TrendFeatureExtractor2D
# from vid2bp.nets.modules.ver2.Amplifier import PleAmplifier
from vid2bp.nets.modules.ver2.PositionalEncoding import PositionalEncoding


class bvp2abp(nn.Module):
    def __init__(self, in_channels, out_channels, target_samp_rate=60, dilation_val=2):
        super(bvp2abp, self).__init__()
        self.in_channel = in_channels
        self.out_channels = out_channels
        self.target_samp_rate = target_samp_rate
        self.dilation_value = dilation_val
        self.p_encoder = PositionalEncoding()
        self.channel_pool = nn.Conv1d(out_channels, 1, kernel_size=1)

        self.linear_model = Linear_module_1D()
        self.DBP = DBP_module(self.in_channel, self.out_channels)
        self.SBP = SBP_module(self.in_channel, self.out_channels)
        # self.detail_feature_extractor = DetailFeatureExtractor(in_channels=self.in_channel,
        #                                                        out_channels=self.out_channels)
        self.detail_feature_extractor2d = DetailFeatureExtractor2D(in_channels=self.in_channel,
                                                                   out_channels=self.out_channels)
        # self.trend_feature_extractor = TrendFeatureExtractor(in_channels=self.in_channel,
        #                                                      out_channels=self.out_channels)
        self.trend_feature_extractor2d = TrendFeatureExtractor2D(in_channels=self.in_channel,
                                                                 out_channels=self.out_channels)
        '''Amplify the small photoplethysmography signal'''
        # self.ple_amplifier = PleAmplifier(in_channels=self.trend_in_channel)

    def forward(self, x, ohe_info):
        batchN, channleN, seqN = x.shape
        ppg_signal = x[:, 0, :].view(batchN, seqN)
        # vpg_signal = x[:, 1, :].view(batchN, seqN)
        # apg_signal = x[:, 2, :].view(batchN, seqN)

        at1, at2 = self.trend_feature_extractor2d(
            x.view(-1, self.in_channel, self.target_samp_rate, seqN // self.target_samp_rate))
        d2_out = self.detail_feature_extractor2d(
            x.view(-1, self.in_channel, self.target_samp_rate, seqN // self.target_samp_rate), at1, at2)

        # ppg_ohe = torch.cat((ppg_signal, ohe_info), dim=-1)
        # d2_ohe = torch.cat((d2_out, ohe_info), dim=-1)
        dbp = torch.abs(self.DBP(ppg_signal))
        sbp = torch.abs(self.SBP(ppg_signal))
        # mbp = (2 * dbp + sbp) / 3
        # sm_height = torch.abs(sbp - mbp)
        # mb_height = torch.abs(mbp - dbp)
        amplitude = torch.abs(sbp - dbp)

        y = self.linear_model(d2_out.view(batchN, -1))
        y_batch_min, y_batch_max = y.min(dim=-1, keepdim=True)[0], y.max(dim=-1, keepdim=True)[0]
        y_norm = (y - y_batch_min) / (y_batch_max - y_batch_min)
        upscaled_y = torch.mul(y_norm, amplitude) + dbp

        return upscaled_y, dbp, sbp, amplitude
