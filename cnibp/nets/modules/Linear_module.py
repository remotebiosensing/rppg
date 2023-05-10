import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchsummary import summary


class Linear_Expansion(nn.Module):
    def __init__(self):
        super(Linear_Expansion, self).__init__()
        self.linear1 = nn.Linear(375, 500)
        self.linear2 = nn.Linear(500, 750)

    def forward(self, feature):
        l1 = self.linear1(feature)
        l2 = self.linear2(l1)
        return l2


class Linear_Projection_DBP(nn.Module):
    def __init__(self):
        super(Linear_Projection_DBP, self).__init__()
        self.linear1 = nn.Linear(375, 25)
        self.linear2 = nn.Linear(25, 15)

    def forward(self, feature):
        l1 = self.linear1(feature)
        l2 = self.linear2(l1)
        return l2


class Linear_Projection_SBP(nn.Module):
    def __init__(self):
        super(Linear_Projection_SBP, self).__init__()
        self.linear1 = nn.Linear(375, 25)
        self.linear2 = nn.Linear(25, 15)

    def forward(self, feature):
        l1 = self.linear1(feature)
        l2 = self.linear2(l1)
        return l2


if __name__ == '__main__':
    model = Linear_Projection_DBP()
    summary(model, (375,), batch_size=2048, device='cpu')
