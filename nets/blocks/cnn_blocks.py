import torch.nn

from nets.blocks.blocks import ConvBlock2D


class cnn_blocks(torch.nn.Module):
    def __init__(self):
        super(cnn_blocks, self).__init__()
        self.cnn_blocks = torch.nn.Sequential(
            ConvBlock2D(3, 16, [5, 5], [1, 1], [2, 2]),
            torch.nn.MaxPool2d((2, 2), stride=(2, 2)),
            ConvBlock2D(16, 32, [3, 3], [1, 1], [1, 1]),
            ConvBlock2D(32, 64, [3, 3], [1, 1], [1, 1]),
            torch.nn.MaxPool2d((2, 2), stride=(2, 2)),
            ConvBlock2D(64, 64, [3, 3], [1, 1], [1, 1]),
            ConvBlock2D(64, 64, [3, 3], [1, 1], [1, 1]),
            torch.nn.MaxPool2d((2, 2), stride=(2, 2)),
            ConvBlock2D(64, 64, [3, 3], [1, 1], [1, 1]),
            ConvBlock2D(64, 64, [3, 3], [1, 1], [1, 1]),
            torch.nn.MaxPool2d((2, 2), stride=(2, 2)),
            ConvBlock2D(64, 64, [3, 3], [1, 1], [1, 1]),
            ConvBlock2D(64, 64, [3, 3], [1, 1], [1, 1]),
            torch.nn.AdaptiveMaxPool2d(1)
        )

    def forward(self, x):
        [batch, channel, length, width, height] = x.shape
        x = x.reshape(batch * length, channel, width, height)
        x = self.cnn_blocks(x)
        return x.reshape(batch, length, -1)

