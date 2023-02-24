import torch
import torch.nn as nn

class TrendFeatureExtractor2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TrendFeatureExtractor2D, self).__init__()
        self.enconv1 = nn.Sequential(
            # nn.Conv1d(in_channels+1, out_channels, kernel_size=3, stride=1, dilation=dilation_val),
            nn.Conv2d(in_channels, out_channels, kernel_size=(5, 1), stride=1),  # channel : 4 -> 8
            nn.BatchNorm2d(out_channels),
            nn.ELU()
        )
        self.enconv2 = nn.Sequential(
            # nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, dilation=dilation_val),
            nn.Conv2d(out_channels, out_channels, kernel_size=(5, 1), stride=1),  # channel : 8 -> 8
            nn.BatchNorm2d(out_channels),
            nn.ELU()
        )
        self.dropout = nn.Dropout2d(0.5)
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels // 8, kernel_size=(5, 5), stride=1, padding=(0, 2)),
            nn.BatchNorm2d(out_channels // 8),
            nn.ELU()
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(out_channels // 8, 1, kernel_size=(5, 5), stride=1, padding=(0, 2)),
            nn.BatchNorm2d(1),
            nn.ELU()
        )

    def forward(self, x):
        x1 = self.enconv1(x)  # (batch, out//2, 358)
        p1 = self.pool(self.dropout(x1))
        x2 = self.enconv2(p1)  # (batch, out//2, 356)
        p2 = self.pool(self.dropout(x2))  # (batch, out//2, 178)
        x3 = self.deconv1(p2)  # (batch, out//2, 89)
        x4 = self.deconv2(x3)  # (batch, out//4, 91)

        # return x8.view(-1, 1, 64) # 45, 8
        return p2, x4
