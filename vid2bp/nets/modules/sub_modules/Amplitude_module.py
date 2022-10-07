import torch.nn as nn


class Amplitude_module(nn.Module):
    def __init__(self):
        super(Amplitude_module, self).__init__()

        self.dbp_linear = nn.Sequential(
            nn.Linear(100, 50),
            nn.Dropout1d(0.5),
            nn.Linear(50, 1),
            nn.ELU()

        )
        self.sbp_linear = nn.Sequential(
            nn.Linear(100, 50),
            nn.Dropout1d(0.5),
            nn.Linear(50, 1),
            nn.ELU()
        )

    def forward(self, d_out):
        dbp = self.dbp_linear(d_out)
        sbp = self.sbp_linear(d_out)

        return dbp, sbp
