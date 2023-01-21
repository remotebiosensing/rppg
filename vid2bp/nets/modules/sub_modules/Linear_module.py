import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class Linear_module_1D(nn.Module):

    def __init__(self):
        super(Linear_module_1D, self).__init__()
        self.linear1 = nn.Linear(95, 180)
        self.linear2 = nn.Linear(180, 270)
        self.linear3 = nn.Linear(270, 360)
        self.elu = nn.ELU()

    def forward(self, feature):
        l1 = self.linear1(feature)
        l2 = self.linear2(l1)
        l3 = self.linear3(l2)
        out = l3

        return out
