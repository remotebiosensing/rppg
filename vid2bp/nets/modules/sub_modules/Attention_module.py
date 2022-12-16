import torch
import torch.nn as nn
from torch import linalg as LA
import matplotlib.pyplot as plt


class Attention_module_1D(nn.Module):
    def __init__(self, in_channels):
        super(Attention_module_1D, self).__init__()
        self.attention = torch.nn.Conv1d(in_channels=in_channels, out_channels=1,
                                         kernel_size=1, stride=1)

    def forward(self, ple_input):
        at = torch.sigmoid(self.attention(ple_input))
        l1norm = LA.norm(at, dim=2)
        at = torch.div(torch.squeeze(at), l1norm)

        return at


