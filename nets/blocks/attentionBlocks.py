import torch
import torch.nn as nn

from ..modules.modules import DAModule


class AttentionBlock_DA(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention = DAModule(in_channels)
        self.conv1x1 = torch.nn.Conv2d(self.attention.inter_channels, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, input):
        mask = self.attention(input)
        mask = torch.sigmoid(mask)
        B, _, H, W = input.shape
        norm = 2 * torch.norm(mask, p=1, dim=(1, 2, 3))
        norm = norm.reshape(B, 1, 1, 1)
        mask = torch.div(mask * H * W, norm)
        mask = self.conv1x1(mask)
        return mask



class AttentionBlock(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention = torch.nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, input):
        mask = self.attention(input)
        mask = torch.sigmoid(mask)
        B, _, H, W = input.shape
        norm = 2 * torch.norm(mask, p=1, dim=(1, 2, 3))
        norm = norm.reshape(B, 1, 1, 1)
        mask = torch.div(mask * H * W, norm)
        return mask