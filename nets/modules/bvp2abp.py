import torch
import torch.nn as nn
from nets.modules.sub_modules.Trend_module import Trend_module_1D
from nets.modules.sub_modules.Amplitude_module import Amplitude_module_1D
from nets.modules.sub_modules.Linear_module import Linear_module_1D


class bvp2abp(nn.Module):
    def __init__(self, in_channels):
        super(bvp2abp, self).__init__()
        self.in_channel = in_channels

        self.trend_model = Trend_module_1D(self.in_channel)
        self.amplitude_model = Amplitude_module_1D(self.in_channel)
        self.linear_model = Linear_module_1D()

        self.conv1 = torch.nn.Sequential(

        )

    def forward(self, ple_input):
        ple_input = torch.reshape(ple_input, (-1, self.in_channel, 360))  # [ batch , channel, size]
        at1, at2 = self.amplitude_model.forward(ple_input)
        t_out = self.trend_model.forward(ple_input, at1, at2)
        l_out = self.linear_model.forward(t_out)

        return l_out
