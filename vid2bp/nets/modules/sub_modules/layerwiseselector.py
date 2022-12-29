import torch.nn as nn
import torch


class LayerWiseSelector(nn.Module):
    def __init__(self, in_layers):
        super(LayerWiseSelector, self).__init__()
        self.in_layer = in_layers
        self.linear1 = nn.Linear(360 * in_layers, 180 * in_layers)
        self.linear2 = nn.Linear(180 * in_layers, 90 * in_layers)
        self.linear3 = nn.Linear(90 * in_layers, 30 * in_layers)
        self.linear4 = nn.Linear(30 * in_layers, in_layers)

    def forward(self, input):  # input: (batch_size, 360 * in_layers)
        l1 = self.linear1(input)
        l2 = self.linear2(l1)
        l3 = self.linear3(l2)
        l4 = self.linear4(l3)
        norm = torch.norm(l4, dim=0)
        p = norm / torch.sum(norm)
        p1, p2 = p[0][0], p[0][1]

        return p1, p2
