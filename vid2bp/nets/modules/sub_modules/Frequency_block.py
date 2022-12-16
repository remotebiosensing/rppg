import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from vid2bp.nets.modules.sub_modules.Attention_module import Attention_module_1D


class frequency_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(frequency_block, self).__init__()
        # self.frequency_feature = torch.fft.fft(norm='forward')
        self.attention = Attention_module_1D(1)

        self.enconv1 = nn.Sequential(  # [batch, in_channels, 360] -> [batch, out_channels, 360]
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm1d(out_channels),
            nn.ELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm1d(out_channels),
            nn.ELU()
        )
        self.enconv2 = nn.Sequential(  # [batch, out_channels, 360] -> [batch, out_channels, 360]
            nn.Conv1d(out_channels, out_channels * 2, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm1d(out_channels * 2),
            nn.Conv1d(out_channels * 2, out_channels * 2, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm1d(out_channels * 2),
            nn.ELU(),
            nn.Conv1d(out_channels * 2, out_channels * 2, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm1d(out_channels * 2),
            nn.Conv1d(out_channels * 2, out_channels * 2, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm1d(out_channels * 2),
            nn.ELU()
        )
        self.dropout = nn.Dropout1d(p=0.5)
        self.max_pool = nn.MaxPool1d(2)
        self.avg_pool = nn.AvgPool1d(2)
        self.linear1 = nn.Linear(50,70)
        self.batch1 = nn.BatchNorm1d(70)
        self.linear2 = nn.Linear(70,90)
        self.batch2 = nn.BatchNorm1d(90)

    def forward(self, ple_input):
        # f = torch.fft.fft(ple_input - torch.mean(ple_input, 2, True), norm='forward')
        # feature =
        f = torch.fft.fft(ple_input - torch.mean(ple_input, 2, True), norm='forward')
        f1 = (abs(f) * (2 / len(f[0, 0, :])))[:, :, :180]
        freq = f1.argsort()[:, :, -50:]
        l1 = self.linear1(torch.squeeze(freq).float())

        l2 = self.linear2(l1)
        # f = (abs(f) * (2 / len(f[0, 0, :])))
        # f1 = self.enconv1(f)
        # p1 = self.max_pool(self.dropout(f1))
        # f2 = self.enconv2(p1)
        # p2 = self.max_pool(self.dropout(f2))

        # f4 = self.max_pool(self.dropout(f3))
        # ifft1 = abs(torch.fft.ifft(f3, norm='forward'))
        # f4 = self.max_pool(self.dropout(ifft1))

        return l2  # [batch, out_channels*2, 90]
    # torch.stack([torch.mean(t_out,dim=1),torch.mean(d_out,dim=1),torch.mean(f_out,dim=1)],dim=1)
