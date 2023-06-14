import torch
import torch.nn as nn
from torchsummary import summary
from cnibp.nets.blocks.conv_blocks import *

class MobileNet(nn.Module):
    def __init__(self, ch_in=3, kernel_size=3):
        super(MobileNet, self).__init__()

        self.config = [
            # t, c, n, k, s

            # t : channel expand ratio
            # c : output channels
            # n : number of blocks (a block contains dw and pw)
            # k : kernel size of depth-wise conv
            # s : stride
            [1, ch_in, 1, kernel_size, 1],  # 이 줄과 [1, 3, , , ] 고정
            [1, ch_in, 1, kernel_size, 1],  # 이 줄은 depthwise conv를 통해 각 input channel을 정보를 학습한다.
            [4, 16, 2, kernel_size, 2],
            [1, 8, 3, kernel_size, 1],
            # [1, 16, 2, 3, 1],
            [1, 4, 4, kernel_size, 1],
            [1, 1, 1, kernel_size, 1]
        ]

        input_channel = ch_in  # network

        # self.stem_conv = conv3(ch_in, 16, stride=2)

        layers = []
        for t, c, n, k, s in self.config:
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(
                    InvertedBlock(ch_in=input_channel, ch_out=c, kernel_size=k, expand_ratio=t, stride=stride))
                input_channel = c

        self.layers = nn.Sequential(*layers)
        self.last_layer = conv1d_1x1(input_channel, ch_out=1)

    def forward(self, x):
        x = self.layers(x)
        x = self.last_layer(x)
        return x


if __name__ == '__main__':
    model = MobileNet(ch_in=3, kernel_size=5)
    summary(model, (3, 750), batch_size=1024, device='cpu')
