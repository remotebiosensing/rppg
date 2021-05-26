import torch
import numpy as np


class TimeDistributed(torch.nn.Module):
    def __init__(self, module, batch_first):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, input_seq):
        assert len(input_seq.size()) > 2

        reshaped_input = input_seq.contiguous().view(-1, input_seq.size(-1))

        output = self.module(reshaped_input)

        if self.batch_first:
            output = output.contiguous().view(input_seq.size(0), -1, output.size(-1))
        else:
            output = output.contiguous().view(-1, input_seq.size(1), output.size(-1))
        return output


class DeepPhys(torch.nn.Module):
    def __init__(self):
        super(DeepPhys, self).__init__()
        self.mask1 = None
        self.mask2 = None

        # Appearance Model
        # padding same = 0
        # padding valid = 1
        self.a_conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.a_Batch_Normalization1 = torch.nn.BatchNorm2d(32)
        self.a_conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.a_Batch_Normalization2 = torch.nn.BatchNorm2d(32)

        self.attention_conv1 = torch.nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.a_avg1 = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.a_Dropout1 = torch.nn.Dropout2d(p=0.50)

        self.a_conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.a_Batch_Normalization3 = torch.nn.BatchNorm2d(64)
        self.a_conv4 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.a_Batch_Normalization4 = torch.nn.BatchNorm2d(64)
        self.a_conv5 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.a_Batch_Normalization5 = torch.nn.BatchNorm2d(64)
        self.a_Dropout2 = torch.nn.Dropout2d(p=0.50)

        self.attention_conv2 = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)

        # Motion Model
        self.m_conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.m_Batch_Normalization1 = torch.nn.BatchNorm2d(32)
        self.m_conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.m_Batch_Normalization2 = torch.nn.BatchNorm2d(32)

        self.m_avg1 = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.m_Dropout1 = torch.nn.Dropout2d(p=0.50)

        self.m_conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.m_Batch_Normalization3 = torch.nn.BatchNorm2d(64)
        self.m_conv4 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.m_Batch_Normalization4 = torch.nn.BatchNorm2d(64)
        self.m_conv5 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.m_Batch_Normalization5 = torch.nn.BatchNorm2d(64)

        self.m_avg2 = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.m_Dropout2 = torch.nn.Dropout2d(p=0.50)

        # foward part flatten
        self.fully_Dropout = torch.nn.Dropout2d(p=0.25)
        self.fully1 = torch.nn.Linear(in_features=64 * 9 * 9, out_features=128, bias=True)
        self.fully2 = torch.nn.Linear(in_features=128, out_features=1, bias=True)

    def forward(self, A, M):
        # Appearance stream
        M1 = torch.tanh(self.m_Batch_Normalization1(self.m_conv1(M)))
        M2 = self.m_Batch_Normalization2(self.m_conv2(M1))
        # M1 = torch.tanh(self.m_conv1(M))
        # M2 = torch.tanh(self.m_conv2(M1))
        # M3 = self.m_Dropout1(M2)

        A1 = torch.tanh(self.a_Batch_Normalization1(self.a_conv1(A)))
        A2 = torch.tanh(self.a_Batch_Normalization2(self.a_conv2(A1)))
        # A1 = torch.tanh(self.a_conv1(A))
        # A2 = torch.tanh(self.a_conv2(A1))
        A3 = self.a_Dropout1(A2)

        # Attention mask1
        mask1 = torch.sigmoid(self.attention_conv1(A3))
        B, _, H, W = A3.shape
        norm = 2 * torch.norm(mask1, p=1, dim=(1, 2, 3))
        norm = norm.reshape(B, 1, 1, 1)
        mask1 = torch.div(mask1 * H * W, norm)
        self.mask1 = mask1

        g1 = torch.tanh(torch.mul(mask1, M2))

        M3 = self.m_Dropout1(g1)
        M4 = self.m_avg1(M3)

        A4 = self.a_avg1(A3)
        # A4 = self.a_Dropout1(A3)

        M5 = torch.tanh(self.m_Batch_Normalization3(self.m_conv3(M4)))
        M6 = self.m_Batch_Normalization4(self.m_conv4(M5))
        # M5 = torch.tanh(self.m_conv3(M4))
        # M6 = torch.tanh(self.m_conv4(M5))
        # M7 = self.m_Dropout1(M6)

        A5 = torch.tanh(self.a_Batch_Normalization3(self.a_conv3(A4)))
        A6 = torch.tanh(self.a_Batch_Normalization4(self.a_conv4(A5)))
        # A5 = torch.tanh(self.a_conv3(A4))
        # A6 = torch.tanh(self.a_conv4(A5))
        A7 = self.a_Dropout2(A6)

        # Attention mask2
        mask2 = torch.sigmoid(self.attention_conv2(A7))
        B, _, H, W = A7.shape
        norm = 2 * torch.norm(mask2, p=1, dim=(1, 2, 3))
        norm = norm.reshape(B, 1, 1, 1)
        mask2 = torch.div(mask2 * H * W, norm)
        self.mask2 = mask2

        g2 = torch.tanh(torch.mul(mask2, M6))

        M7 = self.m_Dropout1(g2)
        M8 = self.m_avg1(M7)

        # Fully connected
        M9 = torch.flatten(M8, start_dim=1)
        M10 = self.fully_Dropout(M9)
        M11 = torch.tanh(self.fully1(M10))
        out = self.fully2(M11)

        return out, self.mask1, self.mask2


class LSTM(torch.nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm1 = torch.nn.LSTM(input_size=4,
                                   hidden_size=128,
                                   num_layers=1,
                                   batch_first=True,
                                   bidirectional=False)

        self.lstm2 = torch.nn.LSTM(input_size=128,
                                   hidden_size=128,
                                   num_layers=1,
                                   batch_first=True,
                                   bidirectional=False)
        self.fc = torch.nn.Linear(128, 1)

    def forward(self, noise):
        layer1 = self.lstm1(noise)
        (LSTM_layer1, _) = self.lstm1(noise)
        # RepeatVector - pytorch
        LSTM_layer1 = LSTM_layer1[:, -1, :].unsqueeze(1).repeat(1, 60, 1)
        # layer2 = self.lstm2(layer1)
        (LSTM_layer2, _) = self.lstm2(LSTM_layer1)
        output = TimeDistributed(torch.tanh(self.fc(LSTM_layer2)), True)

        return output
