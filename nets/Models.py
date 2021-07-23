import torch

from AppearanceModel import AppearanceModel_2D
from MotionModel import motion_model


class Deepphys(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = model
        self.in_channels = 3
        self.out_channels = 32
        self.kernel_size = 3

        self.appearance_model = AppearanceModel_2D(in_channels=self.in_channels, out_channels=self.out_channels * 4,
                                                   kernel_size=self.kernel_size)

        # mot: c-b-c-b-d-a-c-b-c-b-d-a
        # app: c-b-c-b-d-at-a-c-b-c-b-d-at

        # """
        # self.appearance_model = DA_appearance_model(in_channels=self.in_channels, out_channels=self.out_channels,
        #                                          kernel_size=self.kernel_size)
        self.motion_model = motion_model(in_channels=self.in_channels, out_channels=self.out_channels,
                                         kernel_size=self.kernel_size, model=model)

        self.fully = fc()
        # """
        # self.appearance_model = complexAttention(in_channels=self.in_channels, out_channels=self.out_channels,
        #                                             kernel_size=self.kernel_size)
        # self.fully = Compelxfc()

    # def forward(self, appearance_input, motion_input):
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

        # X = torch.Tensor(X).cuda()
        # attention_mask1, attention_mask2,out = self.appearance_model(X)
        # out = self.fully(out)

        return out, attention_mask1, attention_mask2
