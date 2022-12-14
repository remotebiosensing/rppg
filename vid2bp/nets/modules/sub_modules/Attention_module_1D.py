import torch
import torch.nn as nn
from torch import linalg as LA

class Attention_module(nn.Module):
    def __init__(self, in_channels):
        super(Attention_module, self).__init__()
        self.attention = torch.nn.Conv1d(in_channels=in_channels, out_channels=1,
                                         kernel_size=1, stride=1, padding=1)

    def forward(self, ple_input):
        at = torch.sigmoid(self.attention(ple_input))
        l1norm = LA.norm(at, 1)
        at = torch.div(at, l1norm)

        return at
