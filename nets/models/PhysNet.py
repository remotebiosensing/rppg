import torch

from nets.blocks.decoder_blocks import decoder_block
from nets.blocks.encoder_blocks import encoder_block


class PhysNet(torch.nn.Module):
    def __init__(self, frames=32):
        super(PhysNet, self).__init__()
        self.physnet = torch.nn.Sequential(
            encoder_block(),
            decoder_block(),
            torch.nn.AdaptiveMaxPool3d((frames, 1, 1)),  # spatial adaptive pooling
            torch.nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)
        )

    def forward(self, x):
        [batch, channel, length, width, height] = x.shape
        return self.physnet(x).view(-1, length)
