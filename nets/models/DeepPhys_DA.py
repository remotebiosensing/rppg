import torch

from nets.models.sub_models.AppearanceModel import AppearanceModel_DA
from nets.models.sub_models.LinearModel import LinearModel
from nets.models.sub_models.MotionModel import MotionModel


class DeepPhys_DA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 3
        self.out_channels = 32
        self.kernel_size = 3
        self.attention_mask1 = None
        self.attention_mask2 = None

        self.appearance_model = AppearanceModel_DA(in_channels=self.in_channels, out_channels=self.out_channels,
                                                   kernel_size=self.kernel_size)
        self.motion_model = MotionModel(in_channels=self.in_channels, out_channels=self.out_channels,
                                        kernel_size=self.kernel_size)

        self.fully = LinearModel()

    def forward(self, inputs):
        """
        :param appearance_input:
        :param motion_input:
        :return:
        original 2d model
        """
        self.attention_mask1, self.attention_mask2 = self.appearance_model(inputs[0])
        motion_output = self.motion_model(inputs[1], self.attention_mask1, self.attention_mask2)
        out = self.fully(motion_output)

        return out

    def get_attention_mask(self):
        return self.attention_mask1, self.attention_mask2
