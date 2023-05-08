import torch
import torch.nn as nn


# 비디오 데이터를 (b*l, c, h, w) 형태로 바꾸기 위한 Rearrange 모듈 사용
class VideoRearrange(nn.Module):
    def __init__(self):
        super(VideoRearrange, self).__init__()

    def forward(self, x):
        return torch.reshape(x, (x.shape[0] * x.shape[2], x.shape[1], x.shape[3], x.shape[4]))


# LSTM을 이용한 시계열 특징 추출
class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(TimeSeriesLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        output, _ = self.lstm(x)
        return output[:, -1, :]


# 디코더를 이용한 ppg 신호 생성
class PPGDecoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(PPGDecoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.fc(output)
        return output[:, -1, :]


# rPPG 모델
class APNETv3(nn.Module):
    def __init__(self, video_input_size=64, video_hidden_size=128, video_num_layers=2,
                  lstm_input_size=128, lstm_hidden_size=64, lstm_num_layers=2,
                  ppg_input_size=64, ppg_output_size=64, ppg_hidden_size=32, ppg_num_layers=2):
        super(APNETv3, self).__init__()
        self.video_rearrange = VideoRearrange()
        self.video_lstm = nn.LSTM(video_input_size, video_hidden_size, video_num_layers, batch_first=True)
        self.time_series_lstm = TimeSeriesLSTM(lstm_input_size, lstm_hidden_size, lstm_num_layers)
        self.ppg_decoder = PPGDecoder(ppg_input_size, ppg_output_size, ppg_hidden_size, ppg_num_layers)

    def forward(self, video_input):
        # 비디오 데이터 처리
        video_input = self.video_rearrange(video_input)
        output, _ = self.video_lstm(video_input)
        lstm_input = output[:, -1, :]

        # 시계열 특징 추출
        lstm_output = self.time_series_lstm(lstm_input.unsqueeze(0))

        # ppg 신호 생성
        ppg_output = self.ppg_decoder(lstm_output.unsqueeze(0))
        return ppg_output
