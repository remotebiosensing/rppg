import torch.nn

from nets.blocks.blocks import ConvBlock3D


class encoder_block(torch.nn.Module):
    def __init__(self):
        super(encoder_block, self).__init__()
        # self.encoder_block = torch.nn.Sequential(
        #     ConvBlock3D(3, 16, [1, 5, 5], [1, 1, 1], [0, 2, 2]),
        #     torch.nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        #     ConvBlock3D(16, 32, [3, 3, 3], [1, 1, 1], [1, 1, 1]),
        #     ConvBlock3D(32, 64, [3, 3, 3], [1, 1, 1], [1, 1, 1]),
        #     torch.nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)),  # Temporal Halve
        #     ConvBlock3D(64, 64, [3, 3, 3], [1, 1, 1], [1, 1, 1]),
        #     ConvBlock3D(64, 64, [3, 3, 3], [1, 1, 1], [1, 1, 1]),
        #     torch.nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)),  # Temporal Halve
        #     ConvBlock3D(64, 64, [3, 3, 3], [1, 1, 1], [1, 1, 1]),
        #     ConvBlock3D(64, 64, [3, 3, 3], [1, 1, 1], [1, 1, 1]),
        #     torch.nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        #     ConvBlock3D(64, 64, [3, 3, 3], [1, 1, 1], [1, 1, 1]),
        #     ConvBlock3D(64, 64, [3, 3, 3], [1, 1, 1], [1, 1, 1])
        # )
        #, in_channel, out_channel, kernel_size, stride, padding
        self.conv_block1 = ConvBlock3D(3, 16, [1, 5, 5], [1, 1, 1], [0, 2, 2])
        self.max_pool_1 = torch.nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.conv_block2 = ConvBlock3D(16, 32, [3, 3, 3], [1, 1, 1], [1, 1, 1])
        self.conv_block3 = ConvBlock3D(32, 64, [3, 3, 3], [1, 1, 1], [1, 1, 1])
        self.max_pool_2 = torch.nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))  # Temporal Halve
        self.conv_block4 = ConvBlock3D(64, 64, [3, 3, 3], [1, 1, 1], [1, 1, 1])
        self.conv_block5 = ConvBlock3D(64, 64, [3, 3, 3], [1, 1, 1], [1, 1, 1])
        self.max_pool_3 = torch.nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))  # Temporal Halve
        self.conv_block6 = ConvBlock3D(64, 64, [3, 3, 3], [1, 1, 1], [1, 1, 1])
        self.conv_block7 = ConvBlock3D(64, 64, [3, 3, 3], [1, 1, 1], [1, 1, 1])
        self.max_pool_4 = torch.nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.conv_block8 = ConvBlock3D(64, 64, [3, 3, 3], [1, 1, 1], [1, 1, 1])
        self.conv_block9 = ConvBlock3D(64, 64, [3, 3, 3], [1, 1, 1], [1, 1, 1])

    def forward(self, x):
        c1 = self.conv_block1(x)
        m1 = self.max_pool_1(c1)
        c2 = self.conv_block2(m1)
        c3 = self.conv_block3(c2)
        m2 = self.max_pool_2(c3)
        c4  = self.conv_block4(m2)
        c5 = self.conv_block5(c4)
        m3 = self.max_pool_3(c5)
        c6 = self.conv_block6(m3)
        c7 = self.conv_block7(c6)
        m4 = self.max_pool_4(c7)
        c8 = self.conv_block8(m4)
        c9 = self.conv_block9(c8)
        return c9



        # return self.encoder_block(x)
