import torch

from nets.models.sub_models.AppearanceModel import AppearanceModel_2D
from nets.models.sub_models.LinearModel import LinearModel
from nets.models.sub_models.MotionModel import MotionModel


class PPNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1d_1 = torch.nn.Conv1d(1,20,9)
        self.maxpool1d_1 = torch.nn.MaxPool1d(4)
        self.drop_1 = torch.nn.Dropout(0.1)
        self.conv1d_2 = torch.nn.Conv1d(20,20,9)
        self.maxpool1d_2 = torch.nn.MaxPool1d(4)
        self.drop_2 = torch.nn.Dropout(0.1)
        self.lstm_1 = torch.nn.LSTM(13,64)
        self.drop_3 = torch.nn.Dropout(0.1)
        self.lstm_2 = torch.nn.LSTM(64,128)
        self.drop_4 = torch.nn.Dropout(0.1)
        self.flatten = torch.nn.Flatten()
        self.dense = torch.nn.Linear(2560,3)

        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()

    def forward(self, inputs):
        """
        :param inputs:
        inputs[0] : appearance_input
        inputs[1] : motion_input
        :return:
        original 2d model
        """
        x = self.relu(self.conv1d_1(inputs))
        x = self.drop_1(self.maxpool1d_1(x))
        x = self.relu(self.conv1d_2(x))
        x = self.drop_2(self.maxpool1d_2(x))
        x = self.drop_3(self.tanh(self.lstm_1(x)[0]))
        x = self.drop_4(self.tanh(self.lstm_2(x)[0]))
        x = self.flatten(x)
        out = self.dense(x)

        return out
