import torch.nn

from nets.blocks.blocks import ConvBlock3D


class cnn_blocks(torch.nn.Module):
    def __init__(self):
        super(cnn_blocks, self).__init__()
        self.cnn_blocks = torch.nn.Sequential(
            ConvBlock3D(3, 16, [1, 5, 5], [1, 1, 1], [0, 2, 2]),
            torch.nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
            ConvBlock3D(16, 32, [1, 3, 3], [1, 1, 1], [1, 1, 1]),
            ConvBlock3D(32, 64, [1, 3, 3], [1, 1, 1], [1, 1, 1]),
            torch.nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
            ConvBlock3D(64, 64, [1, 3, 3], [1, 1, 1], [1, 1, 1]),
            ConvBlock3D(64, 64, [1, 3, 3], [1, 1, 1], [1, 1, 1]),
            torch.nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
            ConvBlock3D(64, 64, [1, 3, 3], [1, 1, 1], [1, 1, 1]),
            ConvBlock3D(64, 64, [1, 3, 3], [1, 1, 1], [1, 1, 1]),
            torch.nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
            ConvBlock3D(64, 64, [1, 3, 3], [1, 1, 1], [1, 1, 1]),
            ConvBlock3D(64, 64, [1, 3, 3], [1, 1, 1], [1, 1, 1]),
            # torch.nn.AdaptiveMaxPool3d(1)
        )

    def forward(self, x):
        [batch, channel, length, width, height] = x.shape
        # x = x.reshape(batch * length, channel, width, height)
        # x = self.cnn_blocks(x)
        # x = x.reshape(batch,length,-1,1,1)
        x = self.cnn_blocks(x)

        return x


