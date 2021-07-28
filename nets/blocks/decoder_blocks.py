import torch.nn

from nets.blocks.blocks import DeConvBlock3D


class decoder_block(torch.nn.Module):
    def __init__(self):
        super(decoder_block, self).__init__()
        self.decoder_block = torch.nn.Sequential(
            DeConvBlock3D(64, 64, [4, 1, 1], [2, 1, 1], [1, 0, 0]),
            DeConvBlock3D(64, 64, [4, 1, 1], [2, 1, 1], [1, 0, 0])
        )

    def forward(self, x):
        return self.decoder_block(x)
