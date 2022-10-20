import torch
import torch.nn as nn


class Detail_module_1D(nn.Module):
    def __init__(self, in_channels):
        super(Detail_module_1D, self).__init__()
        self.enconv = torch.nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=32,
                      kernel_size=3, stride=1),
            nn.BatchNorm1d(32),
            nn.Conv1d(in_channels=32, out_channels=32,
                      kernel_size=3, stride=1),
            nn.BatchNorm1d(32),
            nn.ELU(),
            # nn.MaxPool1d(2),
            nn.Conv1d(in_channels=32, out_channels=32,
                      kernel_size=3, stride=1),
            nn.BatchNorm1d(32),
            nn.Conv1d(in_channels=32, out_channels=32,
                      kernel_size=3, stride=1),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout1d(0.5),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=32, out_channels=32,
                      kernel_size=3, stride=1),
            nn.BatchNorm1d(32),
            nn.Conv1d(in_channels=32, out_channels=32,
                      kernel_size=3, stride=1),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Conv1d(in_channels=32, out_channels=32,
                      kernel_size=3, stride=1),
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
        self.relu = nn.ReLU()
        self.elu = nn.ELU()

    def forward(self, ple_input, at1, at2):
        enout = self.enconv(ple_input)
        out = self.elu(at1 + enout)
        deout = self.deconv(out)
        out = self.elu(at2 + deout)

        return out
