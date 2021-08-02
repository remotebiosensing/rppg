import torch

from nets.models.sub_models.AppearanceModel import AppearanceModel_2D
from nets.models.sub_models.LinearModel import LinearModel
from nets.models.sub_models.MotionModel import MotionModel_TS


class MTTS(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 3
        self.out_channels = 32
        self.kernel_size = 3
        self.attention_mask1 = None
        self.attention_mask2 = None

        self.appearance_model = AppearanceModel_2D(in_channels=self.in_channels, out_channels=self.out_channels,
                                                   kernel_size=self.kernel_size)
        self.motion_model = MotionModel_TS(in_channels=self.in_channels, out_channels=self.out_channels,
                                           kernel_size=self.kernel_size)

        self.hr_linear_model = LinearModel()
        self.rr_linear_model = LinearModel()

    def forward(self, inputs):
        """
        :param inputs:
        inputs[0] : appearance_input
        inputs[1] : motion_input
        :return:
        original 2d model
        """
        inputs = torch.chunk(inputs,2,dim=1)
        self.attention_mask1, self.attention_mask2 = self.appearance_model(torch.squeeze(inputs[0],1))
        motion_output = self.motion_model(torch.squeeze(inputs[1],1), self.attention_mask1, self.attention_mask2)
        hr_out = self.linear_model(motion_output)
        rr_out = self.linear_model(motion_output)

        return [hr_out,rr_out]

    def get_attention_mask(self):
        return self.attention_mask1, self.attention_mask2
