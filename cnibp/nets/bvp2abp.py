from cnibp.nets.blocks.conv_blocks import conv1d_1x1
from cnibp.nets.modules.Linear_module import *
from cnibp.nets.MobileNet1D import MobileNet
from torchsummary import summary
# from cnibp.nets.modules.sub_modules.Trend_module import Trend_module_1D
# from cnibp.nets.modules.sub_modules.Detail_module import Detail_module_1D
# from cnibp.nets.modules.sub_modules.Linear_module import Linear_module_1D
# from cnibp.nets.modules.sub_modules.Amplitude_module import DBP_module, SBP_module, AMP_module, DBP_2D_Module, \
#     SBP_2D_Module
# from cnibp.nets.modules.sub_modules.residual_block import ResBlock_1D
# from torch.nn.functional import cosine_similarity
# from cnibp.nets.modules.sub_modules.Frequency_block import frequency_block
# from cnibp.nets.modules.ver2.Detail_module import DetailFeatureExtractor2D
# from cnibp.nets.modules.ver2.Trend_module import TrendFeatureExtractor2D
# from cnibp.nets.modules.ver2.Amplifier import PleAmplifier
# from cnibp.nets.modules.ver2.PositionalEncoding import PositionalEncoding


# 맥스풀을 에버리지풀로 바꿔야함 <- 딥파이즈에서 다른 거 합칠 특성 때 이상해진다고
class bvp2abp(nn.Module):
    def __init__(self, in_channels, out_channels=1, target_samp_rate=60):
        super(bvp2abp, self).__init__()
        self.in_channel = in_channels

        # feature extractor with kernel size 3 and 5
        self.small_feature_extractor = MobileNet(ch_in=in_channels, kernel_size=3)
        self.large_feature_extractor = MobileNet(ch_in=in_channels, kernel_size=5)
        # feature squeeze
        self.feature_squeeze = conv1d_1x1(ch_in=2, ch_out=1)

        # feature expansion
        self.expansion = Linear_Expansion()
        # feature projection
        self.projection_dbp = Linear_Projection_DBP()
        self.projection_sbp = Linear_Projection_SBP()
        self.sigmoid = nn.Sigmoid()
        # self.out_channels = out_channels
        # self.target_samp_rate = target_samp_rate
        # self.dilation_value = dilation_val
        # self.p_encoder = PositionalEncoding()
        # self.channel_pool = nn.Conv1d(out_channels, 1, kernel_size=1)

        # self.linear_model = Linear_module_1D()
        # self.DBP = DBP_module()
        # self.SBP = SBP_module()
        # self.AMP = AMP_module()
        # self.DBP_2D = DBP_2D_Module(1, 16)
        # self.SBP_2D = SBP_2D_Module(1, 16)
        # self.trend_model = Trend_module_1D(1)
        # self.detail_model = Detail_module_1D(self.in_channel)
        # self.detail_feature_extractor = DetailFeatureExtractor(in_channels=self.in_channel,
        #                                                        out_channels=self.out_channels)
        # self.detail_feature_extractor2d = DetailFeatureExtractor2D(in_channels=1,
        #                                                            out_channels=self.out_channels)
        # self.trend_feature_extractor = TrendFeatureExtractor(in_channels=self.in_channel,
        #                                                      out_channels=self.out_channels)
        # self.trend_feature_extractor2d = TrendFeatureExtractor2D(in_channels=1,
        #                                                          out_channels=self.out_channels)
        # self.sigmoid = nn.Sigmoid()
        '''Amplify the small photoplethysmography signal'''
        # self.batch_weight_determinator
        # self.ple_amplifier = PleAmplifier(in_channels=self.trend_in_channel)

    def forward(self, x):
        # x.reshape(x.shape[0], 1, -1)
        if torch.Tensor.dim(x) == 2:
            x = x.view(x.shape[0], 1, -1)
        # batchN, channelN, seqN = x.shape
        small_feature = self.small_feature_extractor(x)
        large_feature = self.large_feature_extractor(x)
        feature = torch.cat((small_feature, large_feature), dim=1)
        feature = self.feature_squeeze(feature)

        expanded = self.sigmoid(self.expansion(feature))
        dbp = self.projection_dbp(feature)
        sbp = self.projection_sbp(feature)

        # batchN, seqN = x.shape
        # ppg_signal = x[:, 0:1, :]
        # ppg = x[:, 0, :].view(batchN, seqN, 1)
        # vpg_signal = x[:, 1:2, :]
        # apg_signal = x[:, 2:, :]
        # c_dist = torch.cdist(ppg_signal.view(batchN, seqN, 1),
        #                      ppg.view(batchN, seqN, 1)).view(batchN, 1, seqN, seqN).to('cuda:0')

        # at1, at2 = self.trend_model(x)
        # d_out = self.detail_model.forward(x, self.sigmoid(at1), self.sigmoid(at2))
        # # at1, at2 = self.trend_feature_extractor2d(
        # #     apg_signal.view(-1, 1, self.target_samp_rate, seqN // self.target_samp_rate))
        # # d2_out = self.detail_feature_extractor2d(
        # #     ppg_signal.view(-1, 1, self.target_samp_rate, seqN // self.target_samp_rate), self.sigmoid(at1), self.sigmoid(at2))
        #
        # # ds = self.SBP(apg_signal)
        # # dbp, sbp = torch.split(self.SBP(ppg_signal), 1, dim=-1)
        # # dbp1, sbp1 = torch.split(self.DBP(apg_signal), 1, dim=-1)
        # # dbp = (dbp + dbp1) / 2
        # # sbp = (sbp + sbp1) / 2
        # # sbp = torch.split(ds, 2)
        # # dbp = self.DBP(apg_signal)
        # sbp = torch.abs(self.SBP(d_out.view(batchN, -1)))
        # dbp = torch.abs(self.DBP(d_out.view(batchN, -1)))
        # # sbp = torch.squeeze(self.SBP_2D(c_dist))
        # # dbp = torch.squeeze(self.DBP_2D(c_dist))
        # amplitude = torch.abs(sbp - dbp)
        #
        # y = self.linear_model(d_out.view(batchN, -1))
        # y_batch_min, y_batch_max = y.min(dim=-1, keepdim=True)[0], y.max(dim=-1, keepdim=True)[0]
        # y_norm = (y - y_batch_min) / (y_batch_max - y_batch_min)
        # upscaled_y = torch.mul(y_norm, amplitude) + dbp

        return expanded, dbp, sbp  # , mask_weights


if __name__ == '__main__':
    model = bvp2abp(1)
    summary(model, (750,), batch_size=2048, device='cpu')
