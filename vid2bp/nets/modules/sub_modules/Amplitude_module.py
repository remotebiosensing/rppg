import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class Amplitude_module(nn.Module):
    def __init__(self):
        super(Amplitude_module, self).__init__()

        self.ple_classifier = nn.Sequential(
            nn.Linear(360, 90),
            nn.ELU(),
            nn.Linear(90, 45),
            nn.ELU(),
            nn.Linear(45, 1),
            nn.Sigmoid()
            # nn.Softmax(dim=-1)
        )
        self.classifier1 = nn.Softmax(dim=0)
        self.classifier2 = nn.Softmax(dim=-1)
        self.dbp_linear = nn.Sequential(
            nn.Linear(360, 90),
            nn.Dropout1d(0.5),
            nn.Linear(90, 30),
            nn.Dropout1d(0.5),
            nn.Linear(30, 1)
        )
        self.sbp_linear = nn.Sequential(
            nn.Linear(360, 90),
            nn.Dropout1d(0.5),
            nn.Linear(90, 30),
            nn.Dropout1d(0.5),
            nn.Linear(30, 1)
        )
        self.height = nn.Sequential(
            nn.Linear(360, 90),
            nn.Dropout1d(0.5),
            nn.Linear(90, 30),
            nn.Dropout1d(0.5),
            nn.Linear(30, 1)
        )

        # self.scaler = nn.Sequential(
        #     nn.Linear(360, 90),
        #     nn.Dropout1d(0.5),
        #     nn.Linear(90, 30),
        #     nn.Dropout1d(0.5),
        #     nn.Linear(30, 3)
        # )

    def forward(self, ple_input):
        ple_size = self.ple_classifier(ple_input)
        # dbp = self.dbp_linear(d_out)
        # sbp = self.sbp_linear(d_out)
        # height = self.height(d_out)

        # out = self.scaler(d_out).view(-1, 3)
        # out = self.scaler(d_out)
        # dbp = out[:, 0].view(-1, 1, 1)
        # sbp = out[:, 1].view(-1, 1, 1)
        # height = out[:, 2].view(-1, 1, 1)

        return ple_size
