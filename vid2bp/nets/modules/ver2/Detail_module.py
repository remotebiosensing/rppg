import torch
import torch.nn as nn

class DetailFeatureExtractor2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DetailFeatureExtractor2D, self).__init__()
        self.enconv1 = nn.Sequential(
            # nn.Conv1d(in_channels+1, out_channels, kernel_size=3, stride=1, dilation=dilation_val),
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), stride=1),  # channel : 4 -> 8
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), stride=1, padding=(1, 0)),  # channel : 4 -> 8
            nn.BatchNorm2d(out_channels),
            nn.ELU()
        )
        self.enconv2 = nn.Sequential(
            # nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, dilation=dilation_val),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), stride=1),  # channel : 8 -> 8
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), stride=1, padding=(1, 0)),  # channel : 4 -> 8
            nn.BatchNorm2d(out_channels),
            nn.ELU()
        )
        self.enconv3 = nn.Sequential(
            # nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, dilation=dilation_val),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), stride=1),  # channel : 8 -> 16
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), stride=1, padding=(1, 0)),  # channel : 4 -> 8
            nn.BatchNorm2d(out_channels),
            nn.ELU()
        )
        self.enconv4 = nn.Sequential(
            # nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, dilation=dilation_val),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), stride=1),  # channel : 16 -> 16
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), stride=1, padding=(1, 0)),  # channel : 4 -> 8
            nn.BatchNorm2d(out_channels),
            nn.ELU()
        )
        self.dropout = nn.Dropout2d(0.5)
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels // 2, kernel_size=(3, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(out_channels // 2)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(out_channels // 2, out_channels // 4, kernel_size=(3, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(out_channels // 4),
            nn.ELU()
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(out_channels // 4, out_channels // 8, kernel_size=(3, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(out_channels // 8)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(out_channels // 8, out_channels=1, kernel_size=(3, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(1),
            nn.ELU()
        )
        self.elu = nn.ELU()

    def forward(self, x, at1, at2):
        x1 = self.enconv1(x)  # (batch, out//2, 358)
        x2 = self.enconv2(x1)  # (batch, out//2, 356)
        p1 = self.pool(self.dropout(x2))  # (batch, out//2, 178)
        x3 = self.enconv3(p1)  # (batch, out, 176)
        x4 = self.enconv4(x3)  # (batch, out, 174)
        p2 = self.pool(self.dropout(x4))  # (batch, out, 87)
        x5 = self.deconv1(self.elu(p2 + at1))  # (batch, out//2, 89)
        x6 = self.deconv2(x5)  # (batch, out//4, 91)
        x7 = self.deconv3(x6)  # (batch, out//8, 93)
        x8 = self.deconv4(x7)  # (batch, 1, 95)
        out = self.elu(x8 + at2)
        # return x8.view(-1, 1, 64) # 45, 8
        return out.view(-1, 1, 120)  # 60, 6
        # return x8.view(-1,1,24)
