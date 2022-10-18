import torch


class DepthwiseSeparableConv3d(torch.nn.Module):
    def __init__(self):
        super(DepthwiseSeparableConv3d, self).__init__()
        # N, C, t, H, W
        self.depthwise_conv = torch.nn.Conv3d(in_channels=3, out_channels=3, kernel_size=1, groups=3)
        self.pointwise_conv = torch.nn.Conv3d(in_channels=3, out_channels=1, kernel_size=1)

    def forward(self, x):
        # depthwise convolution
        x = self.depthwise_conv(x)  # (N, C, t, H, W) -> (N, C, t, H, W)
        # average pooling
        x = torch.nn.functional.avg_pool3d(x, kernel_size=1)  # (N, C, t, H, W) -> (N, C, t, H, W)
        # pointwise convolution
        x = self.pointwise_conv(x)  # (N, C, t, H, W) -> (N, 1, t, H, W)
        # softmax
        x = torch.nn.functional.softmax(x, dim=1)
        return x


class AttentionAndPooling(torch.nn.Module):
    def __init__(self):
        super(AttentionAndPooling, self).__init__()

    def forward(self, attmap, segment):
        output_size = segment.shape[2:]
        # attmap * segment
        out = attmap*segment
        # (attmap * segment) + segment
        out = out+segment
        # AdaptiveAvgPool2d
        out = torch.nn.functional.adaptive_avg_pool3d(out, output_size=output_size)
        # adaptive avg pool3d
        # Input : (N, C, t, H, W)
        # Parameter : output_size(txHxW) (tuple)
        # Output : (N, C, t, H, W)
        return out


class ETASubNetBlock(torch.nn.Module):
    def __init__(self):
        super(ETASubNetBlock, self).__init__()
        # define ETA-rPPGNet layers
        self.DepthwiseSeparableConv3d = DepthwiseSeparableConv3d()
        self.AttentionAndPooling = AttentionAndPooling()

    def forward(self, x):
        # depth-wise separable convolution
        attmap = self.DepthwiseSeparableConv3d(x)
        # attention and pooling
        x = self.AttentionAndPooling(attmap, x)
        return x


class Squeeze2(torch.nn.Module):
    def __init__(self):
        super(Squeeze2, self).__init__()

    def forward(self, x):
        x = torch.squeeze(x, dim=-1)
        x = torch.squeeze(x, dim=-1)
        return x


class Unsqueeze2(torch.nn.Module):
    def __init__(self):
        super(Unsqueeze2, self).__init__()

    def forward(self, x):
        x = torch.unsqueeze(x, dim=-1)
        x = torch.unsqueeze(x, dim=-1)
        return x


class STBlock(torch.nn.Module):
    def __init__(self):
        super(STBlock, self).__init__()
        self.conv1 = torch.nn.Conv3d(3, 3, kernel_size=(1, 3, 3))
        self.conv2 = torch.nn.Conv3d(3, 3, kernel_size=(3, 1, 1))

    def forward(self, x):
        x = self.conv1(x)  # (N, C, Block, H, W) -> (N, C, Block, H, W)
        x = self.conv2(x)  # (N, C, Block, H, W) -> (N, C, Block, H, W)
        # 3d average pooling
        x = torch.nn.functional.avg_pool3d(x, kernel_size=(1, 2, 2))  # (N, C, Block, H, W) -> (N, C, Block, H, W)
        return x  # (N, C, Block, H, W)


class TimeDomainAttention(torch.nn.Module):
    def __init__(self):
        super(TimeDomainAttention, self).__init__()
        self.gap3d = torch.nn.AdaptiveAvgPool3d(1)
        self.squeeze2 = Squeeze2()
        self.conv1d = torch.nn.Conv1d(in_channels=3, out_channels=3, kernel_size=5, padding='same')
        self.unsqueeze2 = Unsqueeze2()
        self.activation = torch.nn.Sigmoid()

    def forward(self, d):
        # Global Average Pooling : (N,C,Block,H,W) -> (N,C,Block,1,1)
        m = self.gap3d(d)
        # Squeeze : (N,C,Block,1,1) -> (N,C,Block)
        m = self.squeeze2(m)
        # one-dimensional convolution : (N,C,Block) -> (N,C,Block)
        m = self.conv1d(m)
        # 1 + sigmoid activation : (N,C,Block) -> (N,C,Block)
        m = self.activation(m) + 1
        # Unsqueeze : (N,C,,Block) -> (N,C,Block,1,1)
        m = self.unsqueeze2(m)
        # multiplication : (N,C,Block,1,1) * (N,C,Block,H,W) -> (N,C,Block,H,W)
        d = m * d
        return d


class rPPGgenerator(torch.nn.Module):
    def __init__(self, blocks, length):
        super(rPPGgenerator, self).__init__()
        self.length = length
        self.blocks = blocks
        # (N, C, Block, H, W) -> (N, 3, Block, 1, 1)
        self.AdaptiveAvgPool3d = torch.nn.AdaptiveAvgPool3d((self.blocks, 1, 1))
        # (N, 3, Block, 1, 1) -> (N, 3, Block, 1, 1)
        self.conv3d = torch.nn.Conv3d(3, 1, kernel_size=(1, 1, 1))
        # (N, 3, Block, 1, 1) -> (N, 1, Block, 1, 1)
        self.squeeze2 = Squeeze2()

    def forward(self, x):
        # AdaptiveAvgPool3d
        x = self.AdaptiveAvgPool3d(x)  # (N, C, Block, H, W) -> (N, C, Block, 1, 1)
        # Conv3d
        x = self.conv3d(x)  # (N, C, Block, 1, 1) -> (N, 1, Block, 1, 1)
        # Squeeze
        x = self.squeeze2(x)  # (N, 1, Block, 1, 1) -> (N, 1, Block)
        # Linear interpolation
        x = torch.nn.functional.interpolate(x, size=self.length, mode='linear')  # (N, 1, Block) -> (N, 1, length)

        return x

