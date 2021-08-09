import torch

from nets.blocks.blocks import TSM_Block


class MotionModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        # Motion model
        self.m_conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                       stride=1, padding=1)
        self.m_batch_Normalization1 = torch.nn.BatchNorm2d(out_channels)
        self.m_conv2 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                       stride=1)
        self.m_batch_Normalization2 = torch.nn.BatchNorm2d(out_channels)
        self.m_dropout1 = torch.nn.Dropout2d(p=0.50)

        self.m_avg1 = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.m_conv3 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 2, kernel_size=kernel_size,
                                       stride=1,
                                       padding=1)
        self.m_batch_Normalization3 = torch.nn.BatchNorm2d(out_channels * 2)
        self.m_conv4 = torch.nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels * 2,
                                       kernel_size=kernel_size, stride=1)
        self.m_batch_Normalization4 = torch.nn.BatchNorm2d(out_channels * 2)
        self.m_dropout2 = torch.nn.Dropout2d(p=0.50)
        self.m_avg2 = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, inputs, mask1, mask2):
        M1 = torch.tanh(self.m_batch_Normalization1(self.m_conv1(inputs)))
        M2 = self.m_batch_Normalization2(self.m_conv2(M1))
        # element wise multiplication Mask1
        ones = torch.ones(size=M2.shape).to('cuda')
        g1 = torch.tanh(torch.mul(ones @ mask1, M2))
        M3 = self.m_dropout1(g1)
        # pooling
        M4 = self.m_avg1(M3)
        # g1 = torch.tanh(torch.mul(1 * mask1, M4))
        M5 = torch.tanh(self.m_batch_Normalization3(self.m_conv3(M4)))
        M6 = self.m_batch_Normalization4(self.m_conv4(M5))
        # element wise multiplication Mask2
        g2 = torch.tanh(torch.mul(1 * mask2, M6))
        M7 = self.m_dropout2(g2)
        M8 = self.m_avg2(M7)
        out = torch.tanh(M8)

        return out


class MotionModel_TS(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        # Motion model
        self.m_tsm_conv1 = TSM_Block(in_channels, out_channels, kernel_size, padding='same')
        self.m_tsm_conv2 = TSM_Block(out_channels, out_channels, kernel_size, padding='valid')
        self.m_avg1 = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.m_dropout1 = torch.nn.Dropout2d(p=0.50)
        self.m_tsm_conv3 = TSM_Block(out_channels, out_channels * 2, kernel_size,padding='same')
        self.m_tsm_conv4 = TSM_Block(out_channels * 2, out_channels * 2, kernel_size, padding='valid')
        self.m_avg2 = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.m_dropout2 = torch.nn.Dropout2d(p=0.50)


    def forward(self, inputs, mask1, mask2):
        M1 = torch.tanh(self.m_tsm_conv1(inputs))
        M2 = torch.tanh(self.m_tsm_conv2(M1))
        # element wise multiplication Mask1
        ones = torch.ones(size=M2.shape).to('cuda:9')
        g1 = torch.tanh(torch.mul(ones @ mask1, M2))
        M3 = self.m_avg1(g1)
        # pooling
        M4 = self.m_dropout1(M3)
        # g1 = torch.tanh(torch.mul(1 * mask1, M4))
        M5 = torch.tanh(self.m_tsm_conv3(M4))
        M6 = torch.tanh(self.m_tsm_conv4(M5))
        # element wise multiplication Mask2
        g2 = torch.tanh(torch.mul(1 * mask2, M6))
        M7 = self.m_avg2(g2)
        M8 = self.m_dropout2(M7)
        out = torch.tanh(M8)

        return out
