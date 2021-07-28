import torch

from ..funcs.complexFunctions import complex_matmul


class DAModule(torch.nn.Module):
    def __init__(self, in_channels):
        super(DAModule, self).__init__()
        self.inter_channels = in_channels // 4
        self.conv_p1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, self.inter_channels, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(self.inter_channels),
            torch.nn.ReLU(True)
        )
        self.conv_c1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, self.inter_channels, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(self.inter_channels),
            torch.nn.ReLU(True)
        )
        self.pam = _PositionAttentionModule(self.inter_channels)
        self.cam = _ChannelAttentionModule()
        self.conv_p2 = torch.nn.Sequential(
            torch.nn.Conv2d(self.inter_channels, self.inter_channels, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(self.inter_channels),
            torch.nn.ReLU(True)
        )
        self.conv_c2 = torch.nn.Sequential(
            torch.nn.Conv2d(self.inter_channels, self.inter_channels, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(self.inter_channels),
            torch.nn.ReLU(True)
        )

    def forward(self, x):
        feat_p = self.conv_p1(x)
        feat_p = self.pam(feat_p)
        feat_p = self.conv_p2(feat_p)

        feat_c = self.conv_c1(x)
        feat_c = self.cam(feat_c)
        feat_c = self.conv_c2(feat_c)

        feat_fusion = feat_p + feat_c
        return feat_fusion


class _PositionAttentionModule(torch.nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels):
        super(_PositionAttentionModule, self).__init__()
        self.conv_b = torch.nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = torch.nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = torch.nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = torch.nn.Parameter(torch.zeros(1))
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out


class _ChannelAttentionModule(torch.nn.Module):
    """Channel attention module"""

    def __init__(self):
        super(_ChannelAttentionModule, self).__init__()
        self.beta = torch.nn.Parameter(torch.zeros(1))
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out


class ChannelAttentionModule(torch.nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = torch.nn.Sequential(
            torch.nn.Conv2d(channel, channel // ratio, 1, bias=False),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        # print(avgout.shape)
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(torch.nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = torch.nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(torch.nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = complex_matmul(self.channel_attention(x), x)
        # print('outchannels:{}'.format(out.shape))
        out = complex_matmul(self.spatial_attention(out), out)
        return out


class ResBlock_CBAM(torch.nn.Module):
    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=4):
        super(ResBlock_CBAM, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(places),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1,
                            bias=False),
            torch.nn.BatchNorm2d(places),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1,
                            bias=False),
            torch.nn.BatchNorm2d(places * self.expansion),
        )
        self.cbam = CBAM(channel=places * self.expansion)

        if self.downsampling:
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=in_places, out_channels=places * self.expansion, kernel_size=1,
                                stride=stride, bias=False),
                torch.nn.BatchNorm2d(places * self.expansion)
            )
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        # print(x.shape)
        out = self.cbam(out)
        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out
