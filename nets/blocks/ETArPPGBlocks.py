import torch

class DepthwiseSeparableConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(DepthwiseSeparableConv2d, self).__init__()
        # N, C, H, W
        self.depthwise_conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3,
                                              padding=1, groups=in_channels)
        self.pointwise_conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        # depthwise convolution
        x = self.depthwise_conv(x)
        # average pooling
        x = torch.nn.functional.avg_pool2d(x, kernel_size=2)
        # pointwise convolution
        x = self.pointwise_conv(x)
        # softmax
        x = torch.nn.functional.softmax(x, dim=1)
        return x


class AttentionAndPooling(torch.nn.Module):
    def __init__(self, output_shape):
        super(AttentionAndPooling, self).__init__()
        self.output_shape = output_shape

    def forward(self, attmap, segment):
        # attmap * segment
        out = torch.mul(attmap, segment)
        # (attmap * segment) + segment
        out = torch.sum(out, segment)
        # AdaptiveAvgPool2d
        out = torch.nn.functional.adaptive_avg_pool2d(out, output_size=self.output_shape)
        # adaptive avg pool2d
        # Input : (N, C, H, W)
        # Parameter : output_size(HxW) (tuple)
        # Output : (N, C, output_shape)
        return out


class ETASubNetBlock(torch.nn.Module):
    def __init__(self, output_shape):
        super(ETASubNetBlock, self).__init__()
        self.in_channels = 3
        self.out_channels = 1
        # define ETA-rPPGNet layers
        self.DepthwiseSeparableConv2d = DepthwiseSeparableConv2d(self.in_channels, self.out_channels)
        self.AttentionAndPooling = AttentionAndPooling(output_shape)

    def forward(self, x):
        # depth-wise separable convolution
        attmap = self.DepthwiseSeparableConv2d(x)
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

