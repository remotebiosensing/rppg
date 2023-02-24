import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class DBP_module(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DBP_module, self).__init__()
        self.linear1 = nn.Linear(360, 30)
        self.linear2 = nn.Linear(30, 10)
        self.linear3 = nn.Linear(10, 1)

    def forward(self, x):
        l1 = self.linear1(x)
        l2 = self.linear2(l1)
        l3 = self.linear3(l2)
        return l3


class SBP_module(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SBP_module, self).__init__()
        self.linear1 = nn.Linear(360, 30)
        self.linear2 = nn.Linear(30, 10)
        self.linear3 = nn.Linear(10, 1)

    def forward(self, x):
        l1 = self.linear1(x)
        l2 = self.linear2(l1)
        l3 = self.linear3(l2)
        return l3
