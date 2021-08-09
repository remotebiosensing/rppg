import torch.nn

from nets.blocks.blocks import ConvBlock2D
from nets.blocks.blocks import ConvBlock3D

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
        x = x.view(batch * length, channel, width, height)
        x = self.cnn_blocks(x)
        x = x.view(batch,length,-1,1,1)

        return x

'''
Conv3D 1x3x3(paper architecture)
'''
# class cnn_blocks(torch.nn.Module):
#     def __init__(self):
#         super(cnn_blocks, self).__init__()
#         self.cnn_blocks = torch.nn.Sequential(
#             ConvBlock3D(3, 16, [1, 5, 5], [1, 1, 1], [0, 2, 2]),
#             torch.nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
#             ConvBlock3D(16, 32, [1, 3, 3], [1, 1, 1], [1, 1, 1]),
#             ConvBlock3D(32, 64, [1, 3, 3], [1, 1, 1], [1, 1, 1]),
#             torch.nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
#             ConvBlock3D(64, 64, [1, 3, 3], [1, 1, 1], [1, 1, 1]),
#             ConvBlock3D(64, 64, [1, 3, 3], [1, 1, 1], [1, 1, 1]),
#             torch.nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
#             ConvBlock3D(64, 64, [1, 3, 3], [1, 1, 1], [1, 1, 1]),
#             ConvBlock3D(64, 64, [1, 3, 3], [1, 1, 1], [1, 1, 1]),
#             torch.nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
#             ConvBlock3D(64, 64, [1, 3, 3], [1, 1, 1], [1, 1, 1]),
#             ConvBlock3D(64, 64, [1, 3, 3], [1, 1, 1], [1, 1, 1]),
#             # torch.nn.AdaptiveMaxPool3d(1)
#         )
#
#     def forward(self, x):
#         [batch, channel, length, width, height] = x.shape
#         # x = x.reshape(batch * length, channel, width, height)
#         # x = self.cnn_blocks(x)
#         # x = x.reshape(batch,length,-1,1,1)
#         x = self.cnn_blocks(x)
#
#         return x

