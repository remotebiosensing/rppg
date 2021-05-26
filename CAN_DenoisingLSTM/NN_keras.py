import torch
import numpy as np


class DeepPhys(torch.nn.Module):
    def __init__(self):
        super(DeepPhys, self).__init__()
        self.mask1 = None
        self.mask2 = None

        # Appearance Model
        #padding same = 0
        #padding valid = 1
        self.a_conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0)
        # self.a_Batch_Normalization1 = torch.nn.BatchNorm2d(32)
        self.a_conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        # self.a_Batch_Normalization2 = torch.nn.BatchNorm2d(32)

        self.attention_conv1 = torch.nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.a_avg1 = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.a_Dropout1 = torch.nn.Dropout2d(p=0.25)

        self.a_conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        # self.a_Batch_Normalization3 = torch.nn.BatchNorm2d(64)
        self.a_conv4 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        # self.a_Batch_Normalization4 = torch.nn.BatchNorm2d(64)
        self.a_Dropout2 = torch.nn.Dropout2d(p=0.25)

        self.attention_conv2 = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)

        # Motion Model
        self.m_conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0)
        # self.m_Batch_Normalization1 = torch.nn.BatchNorm2d(32)
        self.m_conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        # self.m_Batch_Normalization2 = torch.nn.BatchNorm2d(32)

        self.m_avg1 = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.m_Dropout1 = torch.nn.Dropout2d(p=0.25)

        self.m_conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        # self.m_Batch_Normalization3 = torch.nn.BatchNorm2d(64)
        self.m_conv4 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        # self.m_Batch_Normalization4 = torch.nn.BatchNorm2d(64)

        self.m_avg2 = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.m_Dropout2 = torch.nn.Dropout2d(p=0.5)

        # foward part flatten
        self.fully1 = torch.nn.Linear(in_features=64 * 9 * 9, out_features=128, bias=True)
        self.fully2 = torch.nn.Linear(in_features=128, out_features=1, bias=True)

    def forward(self, A, M):
        # Appearance stream
        A = torch.tanh(self.a_conv1(A))
        A = torch.tanh(self.a_conv2(A))

        # Attention mask1
        mask1 = torch.sigmoid(self.attention_conv1(A))
        B, _, H, W = A.shape
        norm = 2 * torch.norm(mask1, p=1, dim=(1, 2, 3))
        norm = norm.reshape(B, 1, 1, 1)
        mask1 = torch.div(mask1 * H * W, norm)
        self.mask1 = mask1

        # Pooling
        A = self.a_avg1(A)
        A = self.a_Dropout1(A)

        A = torch.tanh(self.a_conv3(A))
        A = torch.tanh(self.a_conv4(A))

        # Attention mask2
        mask2 = torch.sigmoid(self.attention_conv2(A))
        B, _, H, W = A.shape
        norm = 2 * torch.norm(mask2, p=1, dim=(1, 2, 3))
        norm = norm.reshape(B, 1, 1, 1)
        mask2 = torch.div(mask2 * H * W, norm)
        self.mask2 = mask2

        # Motion stream
        M = torch.tanh(self.m_conv1(M))
        M = torch.tanh(self.m_conv2(M))
        M = torch.mul(M, mask1)

        M = self.m_avg1(M)
        M = self.m_Dropout1(M)

        M = torch.tanh(self.m_conv3(M))
        M = torch.tanh(self.m_conv4(M))
        M = torch.mul(M, mask2)

        M = self.m_avg2(M)
        M = self.m_Dropout2(M)

        # Fully connected
        out = torch.flatten(M, start_dim=1)
        out = torch.tanh(self.fully1(out))
        out = self.m_Dropout2(out)
        out = self.fully2(out)

        return out, self.mask1, self.mask2
