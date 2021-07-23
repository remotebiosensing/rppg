import torch

from nets.AppearanceModel import AppearanceModel_2D, AppearanceModel_DA
from nets.MotionModel import MotionModel
from nets.LinearModel import LinearModel

class Deepphys(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 3
        self.out_channels = 32
        self.kernel_size = 3

        self.appearance_model = AppearanceModel_2D(in_channels=self.in_channels, out_channels=self.out_channels * 4,
                                                   kernel_size=self.kernel_size)
        self.motion_model = MotionModel(in_channels=self.in_channels, out_channels=self.out_channels,
                                        kernel_size=self.kernel_size)

        self.linear_model = LinearModel()

    def forward(self, appearance_input, motion_input):
        """
        :param appearance_input:
        :param motion_input:
        :return:
        original 2d model
        """
        attention_mask1, attention_mask2 = self.appearance_model(appearance_input)
        motion_output = self.motion_model(motion_input, attention_mask1, attention_mask2)
        out = self.linear_model(motion_output)

        return out, attention_mask1, attention_mask2


class Deepphys_DA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 3
        self.out_channels = 32
        self.kernel_size = 3

        self.appearance_model = AppearanceModel_DA(in_channels=self.in_channels, out_channels=self.out_channels,
                                                   kernel_size=self.kernel_size)
        self.motion_model = MotionModel(in_channels=self.in_channels, out_channels=self.out_channels,
                                        kernel_size=self.kernel_size)

        self.fully = LinearModel()

    def forward(self, appearance_input, motion_input):
        """
        :param appearance_input:
        :param motion_input:
        :return:
        original 2d model
        """
        attention_mask1, attention_mask2 = self.appearance_model(appearance_input)
        motion_output = self.motion_model(motion_input, attention_mask1, attention_mask2)
        out = self.fully(motion_output)

        return out, attention_mask1, attention_mask2

