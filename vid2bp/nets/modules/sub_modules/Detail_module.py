# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
# from vid2bp.nets.modules.sub_modules.layerwiseselector import LayerWiseSelector
#
# ''' 미분 값이 들어와서 trend 모듈의 어텐션과 element-wise 곱해줌 '''
#
#
# class Detail_module_1D(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(Detail_module_1D, self).__init__()
#
#         self.upsampler2 = nn.Upsample(scale_factor=2, mode='nearest')
#         self.upsampler4 = nn.Upsample(scale_factor=4, mode='nearest')
#         self.upsampler8 = nn.Upsample(scale_factor=8, mode='nearest')
#         # self.upsampler2 = nn.ConvTranspose1d(out_channels, out_channels, kernel_size=2)
#         # self.upsampler22 = nn.ConvTranspose1d(out_channels, out_channels, kernel_size=2, stride=1)
#         # self.upsampler23 = nn.ConvTranspose1d(out_channels, out_channels, kernel_size=2, stride=2)
#         # self.upsampler23 = nn.ConvTranspose1d(out_channels, out_channels, kernel_size=2, stride=3)
#         # self.upsampler24 = nn.ConvTranspose1d(out_channels, out_channels, kernel_size=2, dilation=1)
#         # self.upsampler25 = nn.ConvTranspose1d(out_channels, out_channels, kernel_size=2, dilation=2)
#         # self.upsampler4 = nn.ConvTranspose1d(out_channels, out_channels, kernel_size=4)
#         # self.upsampler8 = nn.ConvTranspose1d(out_channels, out_channels, kernel_size=8)
#         # self.lws2 = LayerWiseSelector(2)
#         # self.lws3 = LayerWiseSelector(3)
#         # self.lws4 = LayerWiseSelector(4)
#
#         self.enconv1 = nn.Sequential(  # [batch, in_channels, 360] -> [batch, out_channels, 360]
#             nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm1d(out_channels)
#         )
#         self.enconv2 = nn.Sequential(  # [batch, out_channels, 360] -> [batch, out_channels, 360]
#             nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm1d(out_channels),
#             nn.Dropout1d(0.5),
#             nn.AvgPool1d(2)
#         )
#         self.enconv3 = nn.Sequential(  # [batch, out_channel, 180] -> [batch, out_channel*2, 180]
#             nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm1d(out_channels),
#             nn.Dropout1d(0.5),
#             nn.AvgPool1d(2)
#         )
#         self.enconv4 = nn.Sequential(  # [batch, out_channels*2, 360] -> [batch, out_channels*2, 360]
#             nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm1d(out_channels),
#             nn.AvgPool1d(2)
#         )
#         self.channel_selection = nn.Sequential(
#             nn.AdaptiveAvgPool1d(out_channels),
#             nn.Linear(out_channels, out_channels // 4),
#             nn.Linear(out_channels // 4, out_channels // 4),
#         )
#         self.channel_attention = nn.Sequential(
#             nn.Conv1d(out_channels, 1, kernel_size=1, stride=1),
#             nn.BatchNorm1d(1),
#         )
#         self.dropout = nn.Dropout1d(p=0.5)
#         self.max_pool = nn.MaxPool1d(2)
#         self.elu = nn.ELU()
#         self.relu = nn.ReLU()
#
#     def forward(self, ple):  # ple_diff: [batch, 1, 360]
#         # d1 = self.enconv1(ple)  # [batch, 1, 360] -> [batch, 16, 360]
#         # a1 = self.channel_attention(d1)
#         # p1, p2 = self.lws2(torch.cat((ple, a1), dim=-1))  # [batch, 1, 720]
#         # d2 = p1*ple + self.enconv2(a1)  # [batch, 16, 360] -> [batch, 16, 360]
#         # a2 = self.channel_attention(d2)
#         # p1, p2 = self.lws2(torch.cat((ple, a2), dim=-1))  # [batch, 1, 1080]
#         # d3 = p1*a1 + self.enconv3(a2)  # [batch, 16, 360] -> [batch, 16, 360]
#         # a3 = self.channel_attention(d3)
#         # p1, p2 = self.lws2(torch.cat((ple, a3), dim=-1))
#         # d4 = p1*a2 + self.enconv4(a3)  # [batch, 16, 360] -> [batch, 16, 360]
#         # a4 = self.channel_attention(d4)
#         # # p1, p2 = self.lws2(torch.cat((a3, a4), dim=-1))
#         # out = a4
#
#         # d1 = self.enconv1(ple)
#         # d2 = self.enconv2(d1)
#         # d3 = self.enconv3(d2)
#         # d4 = self.enconv4(d3)
#         #
#         # d1 = self.enconv1(ple)  # [batch, 1, 360] -> [batch, 16, 360]
#         # d2 = self.enconv2(d1)  # [batch, 16, 360] -> [batch, 16, 360]
#         # d3 = self.enconv3(d2)  # [batch, 16, 360] -> [batch, 16, 360]
#         # d4 = self.enconv4(d3)  # [batch, 16, 360] -> [batch, 16, 360]
#         # out = self.channel_attention(d1) + self.channel_attention(d2) + self.channel_attention(d3) + self.channel_attention(d4)
#
#         d1 = self.enconv1(ple)  # [batch, out_channels, 360]
#         d2 = self.enconv2(d1)  # [batch, out_channels, 360]
#         d_fuse1 = (d1 + self.upsampler2(d2)) * self.upsampler2(d2)
#         d3 = self.enconv3(d2)  # [batch, out_channels*2, 180]
#         d_fuse2 = (d_fuse1 + self.upsampler4(d3)) * self.upsampler4(d3)  # + t_fuse2
#         d4 = self.enconv4(d3)  # [batch, out_channels*2, 180]
#         d_fuse3 = (d_fuse2 + self.upsampler8(d4)) * self.upsampler8(d4)  # + t_fuse3
#         # out = self.channel_attention(d_fuse3)  # [batch, 1, 360]
#         out = self.channel_attention(self.upsampler2(d2) + self.upsampler4(d3) + self.upsampler8(d4))  # [batch, 1, 360]
#         # plt.plot(self.upsampler2(d2)[0][0].detach().cpu(), label='upsampler2')
#         # plt.plot(self.upsampler4(d3)[0][0].detach().cpu(), label='upsampler4')
#         # plt.plot(self.upsampler8(d4)[0][0].detach().cpu(), label='upsampler8')
#         # plt.legend()
#         # plt.show()
#         return out#, d_fuse1[0], d_fuse2[0], d_fuse3[0]

import torch
import torch.nn as nn


class Detail_module_1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Detail_module_1D, self).__init__()
        self.enconv = torch.nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm1d(32),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm1d(32),
            nn.ELU(),
            # nn.MaxPool1d(2),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm1d(32),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout1d(0.5),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm1d(32),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm1d(32),
            nn.Conv1d(in_channels=32, out_channels=32,
                      kernel_size=3, stride=1),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout1d(0.5),
            nn.MaxPool1d(2)
        )
        self.deconv = torch.nn.Sequential(
            nn.ConvTranspose1d(in_channels=32, out_channels=32,
                               kernel_size=3, stride=1),
            nn.BatchNorm1d(32),
            nn.ConvTranspose1d(in_channels=32, out_channels=32,
                               kernel_size=3, stride=1),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.ConvTranspose1d(in_channels=32, out_channels=32,
                               kernel_size=3, stride=1),
            nn.BatchNorm1d(32),
            nn.ConvTranspose1d(in_channels=32, out_channels=16,
                               kernel_size=3, stride=1),
            nn.BatchNorm1d(16),
            nn.ELU(),
            nn.ConvTranspose1d(in_channels=16, out_channels=16,
                               kernel_size=3, stride=1),
            nn.BatchNorm1d(16),
            nn.ConvTranspose1d(in_channels=16, out_channels=16,
                               kernel_size=3, stride=1),
            nn.BatchNorm1d(16),
            nn.ELU(),
            nn.ConvTranspose1d(in_channels=16, out_channels=16,
                               kernel_size=3, stride=1),
            nn.BatchNorm1d(16),
            nn.ConvTranspose1d(in_channels=16, out_channels=1,
                               kernel_size=3, stride=1),
            nn.BatchNorm1d(1),
            nn.ELU()
        )
        self.cycle_conv = torch.nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=out_channels, kernel_size=3,)
        )
        self.relu = nn.ReLU()
        self.elu = nn.ELU()

    def forward(self, ple_input):
        # _, channelN, _ = ple_input.shape
        enout = self.enconv(ple_input)
        out = self.elu(enout)
        deout = self.deconv(out)
        out = self.elu(deout)

        return out
