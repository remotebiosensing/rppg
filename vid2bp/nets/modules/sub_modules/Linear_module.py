import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class Linear_module_1D(nn.Module):

    def __init__(self):
        super(Linear_module_1D, self).__init__()
        self.linear1 = nn.Linear(120, 180)
        self.linear2 = nn.Linear(180, 240)
        self.linear3 = nn.Linear(240, 300)
        self.linear4 = nn.Linear(300, 360)
        self.sigmoid = nn.Sigmoid()
        self.elu = nn.ELU()

    def forward(self, feature):
        l1 = self.linear1(feature)
        l2 = self.linear2(l1)
        l3 = self.linear3(l2)
        l4 = self.linear4(l3)
        # out = l3
        out = self.sigmoid(l4)

        return out
