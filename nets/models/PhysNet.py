import torch

from nets.blocks.decoder_blocks import decoder_block
from nets.blocks.encoder_blocks import encoder_block
from nets.blocks.cnn_blocks import cnn_blocks
from nets.modules.modules import ConvLSTM


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

class PhysNet_2DCNN_LSTM(torch.nn.Module):
    def __init__(self, frame=32):
        super(PhysNet_2DCNN_LSTM, self).__init__()
        self.physnet_lstm = torch.nn.ModuleDict({
            'cnn_blocks' : cnn_blocks(),
            'cov_lstm' : ConvLSTM(64,[1,1,64],(1,1),num_layers=3, batch_first=True, bias=True, return_all_layers=False),
            'cnn_flatten' : torch.nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)
        })

    def forward(self, x):
        [batch, channel, length, width, height] = x.shape
        x = self.physnet_lstm['cnn_blocks'](x)
        x,_ = self.physnet_lstm['cov_lstm'](x)
        x = torch.permute(x[0], (0, 2, 1, 3, 4))
        x = self.physnet_lstm['cnn_flatten'](x)
        return x.view(-1, length)