import torch
import torch.nn as nn
import torch.nn.functional as F


class skin_based_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(skin_based_Attention, self).__init__()
        self.GAP = nn.AdaptiveAvgPool3d()
