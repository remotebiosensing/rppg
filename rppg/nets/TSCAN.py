import torch
import torch.nn as nn
from rppg.nets.DeepPhys import AppearanceModel, MotionModel, LinearModel
from torchsummary import summary

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

class TSM(nn.Module):
    def __init__(self, time_length=180, fold_div=3):
        super().__init__()
        self.time_length = time_length
        self.fold_div = fold_div

    def forward(self, input):
        B, C, H, W = input.shape
        input = input.view(-1, self.time_length, C, H, W)

        fold = C // self.fold_div
        last_fold = C - (self.fold_div - 1) * fold

        out1, out2, out3 = torch.split(input, [fold, fold, last_fold], dim=2)

        up_out1 = torch.cat((torch.zeros((B//self.time_length, 1, fold, H, W)).to(device), out1[:, 1:, :, :, :]), dim=1)
        down_out2 = torch.cat((out2[:, :-1, :, :, :], torch.zeros((B//self.time_length, 1, fold, H, W)).to(device)), dim=1)
        bidirection_out = torch.cat((up_out1, down_out2, out3), dim=2).view(B, C, H, W)

        return bidirection_out


class MotionBranch(MotionModel):
    def __init__(self, in_channels, out_channels, kernel_size, time_length=180):
        super().__init__(in_channels, out_channels, kernel_size)
        self.tsm = TSM(time_length=time_length, fold_div=3)

    def forward(self, inputs, mask1, mask2):
        m1 = torch.tanh(self.m_conv1(self.tsm(inputs)))
        m2 = torch.tanh(self.m_conv2(self.tsm(m1)))

        p1 = self.m_avg1(m2 * mask1)
        d1 = self.m_dropout1(p1)

        m3 = torch.tanh(self.m_conv3(self.tsm(d1)))
        m4 = torch.tanh(self.m_conv4(self.tsm(m3)))

        p2 = self.m_avg2(m4 * mask2)
        d2 = self.m_dropout2(p2)

        return d2


class TSCAN(nn.Module):
    def __init__(self, timelength=180):
        super(TSCAN, self).__init__()
        self.in_channels = 3
        self.out_channels = 32
        self.kernel_size = 3
        self.time_length = timelength
        self.attention_mask1 = None
        self.attention_mask2 = None

        self.appearance_model = AppearanceModel(in_channels=self.in_channels, out_channels=self.out_channels,
                                                kernel_size=self.kernel_size)
        self.motion_model = MotionBranch(in_channels=self.in_channels, out_channels=self.out_channels,
                                         kernel_size=self.kernel_size, time_length=self.time_length)
        self.linear_model = LinearModel(16384)

    def forward(self, inputs):
        averaged_frames = inputs[0]
        S, C, H, W = averaged_frames.shape
        averaged_frames = averaged_frames.view(-1, self.time_length, C, H, W)
        averaged_frames = torch.mean(averaged_frames, dim=1, keepdim=True).expand(-1, self.time_length, -1, -1, -1).reshape(S, C, H, W)

        self.attention_mask1, self.attention_mask2 = self.appearance_model(averaged_frames)
        motion_output = self.motion_model(inputs[1], self.attention_mask1, self.attention_mask2)

        out = self.linear_model(motion_output)

        return out


# if __name__ == '__main__':
#     model = TSCAN().to('cpu')
#     summary(model=model, input_size=(360, 3, 36, 36), device='cpu')
