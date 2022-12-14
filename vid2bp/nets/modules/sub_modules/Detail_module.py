import torch
import torch.nn as nn
import matplotlib.pyplot as plt

''' 미분 값이 들어와서 trend 모듈의 어텐션과 element-wise 곱해줌 '''


class Detail_module_1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Detail_module_1D, self).__init__()

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
        self.enconv4 = nn.Sequential(  # [batch, out_channels*2, 360] -> [batch, out_channels*2, 360]
            nn.Conv1d(out_channels * 2, out_channels * 2, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm1d(out_channels * 2),
            nn.ELU()
        )
        self.dropout = nn.Dropout1d(p=0.5)
        self.max_pool = nn.MaxPool1d(2)
        self.elu = nn.ELU()

    def forward(self, ple_diff):  # ple_diff: [batch, 2, 360]
        d1 = self.enconv1(ple_diff)  # [batch, out_channels, 360]
        d2 = self.enconv2(d1)  # [batch, out_channels, 360]
        # mul1 = torch.mul(d2, at1.unsqueeze(1))
        p1 = self.max_pool(self.dropout(d1+d2))  # [batch, out_channels, 180]

        d3 = self.enconv3(p1)  # [batch, out_channels*2, 180]
        d4 = self.enconv4(d3)  # [batch, out_channels*2, 180]
        # mul2 = torch.mul(d4, at2.unsqueeze(1))
        p2 = self.max_pool(self.dropout(d3+d4))  # [batch, out_channels*2, 90]

        return p2  # [batch, out_channels*2, 90]
