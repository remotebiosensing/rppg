import torch
import torch.nn as nn

from blocks import TSM_Block


class MotionBlock_2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, model):
        super().__init__()
        self.model = model

        self.m_conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.m_batch1 = torch.nn.BatchNorm2d(out_channels)
        self.m_conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.m_batch2 = torch.nn.BatchNorm2d(out_channels)
        self.m_drop3 = torch.nn.Dropout2d(0.5)
        self.m_avg3 = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, inputs, mask):
        m = self.m_tsm1(inputs)
        m = torch.tanh(m)
        m = self.m_tsm2(m)

        m = torch.mul(m, mask)
        m = torch.tanh(m)

        m = self.m_drop3(m)
        m = self.m_avg3(m)
        return m


class MotionBlock_TS(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, model):
        super().__init__()
        self.model = model

        self.m_tsm1 = TSM_Block(in_channels, out_channels, kernel_size)
        self.m_tsm2 = TSM_Block(out_channels, out_channels, kernel_size)
        self.m_drop3 = torch.nn.Dropout2d(0.5)
        self.m_avg3 = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, inputs, mask):
        m = self.m_conv1(inputs)
        m = self.m_batch1(m)
        m = torch.tanh(m)

        m = self.m_conv2(m)
        m = self.m_batch2(m)

        m = torch.mul(m, mask)
        m = torch.tanh(m)

        m = self.m_drop3(m)
        m = self.m_avg3(m)
        return m
