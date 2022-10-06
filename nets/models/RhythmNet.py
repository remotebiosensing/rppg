import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as models
import ssl

ssl._create_default_https_context = ssl._create_stdlib_context

'''
Backbone CNN for RhythmNet model is a RestNet-18
'''


class RhythmNet(nn.Module):
    def __init__(self):
        super(RhythmNet, self).__init__()

        # resnet o/p -> bs x 1000
        # self.resnet18 = resnet18(pretrained=False)
        resnet = models.resnet18(pretrained=False)
        modules = list(resnet.children())[:-1]

        self.resnet18 = nn.Sequential(*modules)
        # The resnet average pool layer before fc
        # self.avgpool = nn.AvgPool2d((10, 1))
        self.resnet_linear = nn.Linear(512, 1000)
        self.fc_regression = nn.Linear(1000, 1)
        self.gru_fc_out = nn.Linear(1000, 1)
        self.rnn = nn.GRU(input_size=1000, hidden_size=1000, num_layers=1)
        # self.fc = nn.Linear(config.GRU_TEMPORAL_WINDOW, config.GRU_TEMPORAL_WINDOW)

    def forward(self, st_maps, target):
        batched_output_per_clip = []
        gru_input_per_clip = []
        hr_per_clip = []

        # Need to have so as to reflect a batch_size = 1 // if batched then comment out
        st_maps = st_maps.unsqueeze(0)
        for t in range(st_maps.size(1)):
            # with torch.no_grad():
            x = self.resnet18(st_maps[:, t, :, :, :])
            # collapse dimensions to BSx512 (resnet o/p)
            x = x.view(x.size(0), -1)
            # output dim: BSx1 and Squeeze sequence length after completing GRU step
            x = self.resnet_linear(x)
            # Save CNN features per clip for the GRU
            gru_input_per_clip.append(x.squeeze(0))

            # Final regression layer for CNN features -> HR (per clip)
            x = self.fc_regression(x)
            # normalize HR by frame-rate: 25.0 for VIPL
            x = x * 25.0
            batched_output_per_clip.append(x.squeeze(0))
            # input should be (seq_len, batch, input_size)

        # the features extracted from the backbone CNN are fed to a one-layer GRU structure.
        regression_output = torch.stack(batched_output_per_clip, dim=0).permute(1, 0)

        # Trying out GRU in addition to the regression now.
        gru_input = torch.stack(gru_input_per_clip, dim=0)
        gru_output, h_n = self.rnn(gru_input.unsqueeze(1))
        # gru_output = gru_output.squeeze(1)
        for i in range(gru_output.size(0)):
            hr = self.gru_fc_out(gru_output[i, :, :])
            hr_per_clip.append(hr.flatten())

        gru_output_seq = torch.stack(hr_per_clip, dim=0).permute(1, 0)
        # return output_seq, gru_output.squeeze(0), fc_out
        return regression_output, gru_output_seq.squeeze(0)[:6]

    def name(self):
        return "RhythmNet"


if __name__ == '__main__':
    # cm = RhythmNet()
    # img = torch.rand(3, 28, 28)
    # target = torch.randint(1, 20, (5, 5))
    # x = cm(img)
    # print(x)
    resnet18 = models.resnet18(pretrained=False)
    print(resnet18)