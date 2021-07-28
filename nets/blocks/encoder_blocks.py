import torch.nn

from nets.blocks.blocks import ConvBlock3D


class encoder_block(torch.nn.Module):
    def __init__(self):
        super(encoder_block, self).__init__()
        self.encoder_block = torch.nn.Sequential(
            ConvBlock3D(3, 16, [1, 5, 5], [1, 1, 1], [0, 2, 2]),
            torch.nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
            ConvBlock3D(16, 32, [3, 3, 3], [1, 1, 1], [1, 1, 1]),
            ConvBlock3D(32, 64, [3, 3, 3], [1, 1, 1], [1, 1, 1]),
            torch.nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)),  # Temporal Halve
            ConvBlock3D(64, 64, [3, 3, 3], [1, 1, 1], [1, 1, 1]),
            ConvBlock3D(64, 64, [3, 3, 3], [1, 1, 1], [1, 1, 1]),
            torch.nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)),  # Temporal Halve
            ConvBlock3D(64, 64, [3, 3, 3], [1, 1, 1], [1, 1, 1]),
            ConvBlock3D(64, 64, [3, 3, 3], [1, 1, 1], [1, 1, 1]),
            torch.nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
            ConvBlock3D(64, 64, [3, 3, 3], [1, 1, 1], [1, 1, 1]),
            ConvBlock3D(64, 64, [3, 3, 3], [1, 1, 1], [1, 1, 1])
        )
        # self.conv_block1 = ConvBlock3D(3, 16, [1, 5, 5], [1, 1, 1], [0, 2, 2])
        # self.max_pool_1 = torch.nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        # self.conv_block2 = ConvBlock3D(16, 32, [3, 3, 3], [1, 1, 1], [1, 1, 1])
        # self.conv_block3 = ConvBlock3D(32, 64, [3, 3, 3], [1, 1, 1], [1, 1, 1])
        # self.max_pool_2 = torch.nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))  # Temporal Halve
        # self.conv_block4 = ConvBlock3D(64, 64, [3, 3, 3], [1, 1, 1], [1, 1, 1])
        # self.conv_block5 = ConvBlock3D(64, 64, [3, 3, 3], [1, 1, 1], [1, 1, 1])
        # self.max_pool_3 = torch.nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))  # Temporal Halve
        # self.conv_block6 = ConvBlock3D(64, 64, [3, 3, 3], [1, 1, 1], [1, 1, 1])
        # self.conv_block7 = ConvBlock3D(64, 64, [3, 3, 3], [1, 1, 1], [1, 1, 1])
        # self.max_pool_4 = torch.nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        # self.conv_block8 = ConvBlock3D(64, 64, [3, 3, 3], [1, 1, 1], [1, 1, 1])
        # self.conv_block9 = ConvBlock3D(64, 64, [3, 3, 3], [1, 1, 1], [1, 1, 1])

    def forward(self, x):
        return self.encoder_block(x)
