import torch
import torch.nn as nn


class Detail_module_1D(nn.Module):
    def __init__(self, in_channels):
        super(Detail_module_1D, self).__init__()
        self.in_channel = in_channels
        self.enconv = torch.nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=32,
                      kernel_size=3, stride=1),
            nn.BatchNorm1d(32),
            nn.Conv1d(in_channels=32, out_channels=32,
                      kernel_size=3, stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # nn.MaxPool1d(2),
            nn.Conv1d(in_channels=32, out_channels=32,
                      kernel_size=3, stride=1),
            nn.BatchNorm1d(32),
            nn.Conv1d(in_channels=32, out_channels=32,
                      kernel_size=3, stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=32, out_channels=32,
                      kernel_size=3, stride=1),
            nn.BatchNorm1d(32),
            nn.Conv1d(in_channels=32, out_channels=32,
                      kernel_size=3, stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # nn.MaxPool1d(2)
            nn.Conv1d(in_channels=32, out_channels=32,
                      kernel_size=3, stride=1),
            nn.BatchNorm1d(32),
            nn.Conv1d(in_channels=32, out_channels=32,
                      kernel_size=3, stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
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
            nn.ConvTranspose1d(in_channels=16, out_channels=8,
                               kernel_size=3, stride=1),
            nn.BatchNorm1d(8),
            nn.ELU(),
            nn.ConvTranspose1d(in_channels=8, out_channels=8,
                               kernel_size=3, stride=1),
            nn.BatchNorm1d(8),
            nn.ConvTranspose1d(in_channels=8, out_channels=1,
                               kernel_size=3, stride=1),
            nn.BatchNorm1d(1),
            nn.ELU()
        )

    # TODO FORWARD안에 FEATURE 앞에다가 DATALOADER(__GETITEM__())에서 얻은 크기 정보 추가
    def forward(self, ple_input):
        at1 = self.enconv(ple_input)
        at2 = self.deconv(at1)
        return at1, at2
