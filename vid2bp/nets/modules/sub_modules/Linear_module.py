import torch
import torch.nn as nn


class Linear_module_1D(nn.Module):
    def __init__(self):
        super(Linear_module_1D, self).__init__()
        # self.adaptivepool1 = nn.AdaptiveMaxPool1d(180)
        # self.adaptivepool2 = nn.AdaptiveMaxPool1d(270)
        # self.adaptivepool3 = nn.AdaptiveMaxPool1d(360)
        self.linear1 = nn.Linear(100, 180)
        self.linear2 = nn.Linear(180, 270)
        self.linear3 = nn.Linear(270, 360)

    def forward(self, input):
        # l1 = self.adaptivepool1(input)
        # l2 = self.adaptivepool2(l1)
        # l3 = self.adaptivepool3(l2)
        l1 = self.linear1(input)
        l2 = self.linear2(l1)
        l3 = self.linear3(l2)

        return l3
