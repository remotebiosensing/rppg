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
from vid2bp.nets.modules.ver2.Detail_module import DetailFeatureExtractor
from vid2bp.nets.modules.ver2.Trend_module import TrendFeatureExtractor
# from vid2bp.nets.modules.ver2.Amplifier import PleAmplifier
from vid2bp.nets.modules.ver2.PositionalEncoding import PositionalEncoding


class bvp2abp(nn.Module):
    def __init__(self, in_channels, out_channels, target_samp_rate=60, dilation_val=2):
        super(bvp2abp, self).__init__()
        self.in_channel = in_channels
        self.out_channels = out_channels
        self.dilation_value = dilation_val
        self.p_encoder = PositionalEncoding()
        self.channel_pool = nn.Conv1d(out_channels, 1, kernel_size=1)

        self.linear_model = Linear_module_1D()
        self.DBP = DBP_module()
        self.SBP = SBP_module()
        self.detail_feature_extractor = DetailFeatureExtractor(in_channels=self.in_channel,
                                                               out_channels=self.out_channels)
        # self.trend_feature_extractor = TrendFeatureExtractor(in_channels=self.in_channel,
        #                                                      out_channels=self.out_channels)
        '''Amplify the small photoplethysmography signal'''
        # self.ple_amplifier = PleAmplifier(in_channels=self.trend_in_channel)

    def forward(self, x):
        batchN, channleN, seqN = x.shape
        # ppg_signal = x[:, 0, :].view(batchN, seqN)
        # vpg_signal = x[:, 1, :].view(batchN, seqN)
        # apg_signal = x[:, 2, :].view(batchN, seqN)
        # ppg_batch_min, ppg_batch_max = ppg_signal.min(dim=-1, keepdim=True)[0], ppg_signal.max(dim=-1, keepdim=True)[0]
        # vpg_batch_min, vpg_batch_max = vpg_signal.min(dim=-1, keepdim=True)[0], vpg_signal.max(dim=-1, keepdim=True)[0]
        # apg_batch_min, apg_batch_max = apg_signal.min(dim=-1, keepdim=True)[0], apg_signal.max(dim=-1, keepdim=True)[0]
        # ppg_new_min, ppg_new_max = 1, 3
        # vpg_new_min, vpg_new_max = 1, 3
        # apg_new_min, apg_new_max = 1, 3
        # ppg_norm = (ppg_signal - ppg_batch_min) / (ppg_batch_max - ppg_batch_min)*(ppg_new_max - ppg_new_min) + ppg_new_min
        # vpg_norm = (vpg_signal - vpg_batch_min) / (vpg_batch_max - vpg_batch_min)*(vpg_new_max - vpg_new_min) + vpg_new_min
        # apg_norm = (apg_signal - apg_batch_min) / (apg_batch_max - apg_batch_min)*(apg_new_max - apg_new_min) + apg_new_min
        # x[:, 0, :] = ppg_norm
        # x[:, 1, :] = vpg_norm
        # x[:, 2, :] = apg_norm
        x_pos = self.p_encoder(x)
        # t_out = self.trend_feature_extractor(x_pos)
        d_out = self.detail_feature_extractor(x_pos)
        # p1 = self.channel_pool(d_out)
        # p2 = self.channel_pool(ds_in)
        # l_in, ds_in = self.channel_pool(f_out)
        dbp = self.DBP(d_out)
        sbp = self.SBP(d_out)
        amplitude = torch.abs(sbp - dbp)

        y = self.linear_model(d_out.view(batchN, -1))
        y_batch_min, y_batch_max = y.min(dim=-1, keepdim=True)[0], y.max(dim=-1, keepdim=True)[0]
        y_norm = (y - y_batch_min) / (y_batch_max - y_batch_min)
        upscaled_y = torch.mul(y_norm, amplitude) + dbp

        return upscaled_y, dbp, sbp
