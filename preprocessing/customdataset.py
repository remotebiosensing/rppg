import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, x_data, y_data):
        # 들어온 x는 tensor형태로 변환
        self.x_data = torch.FloatTensor(x_data)
        # tensor data의 형태는 (배치사이즈, 채널사이즈, 이미지 너비, 높이)의 형태임
        # 따라서 들어온 데이터의 형식을 permute함수를 활용하여 바꾸어주어야함.
        # self.x_data = self.x_data.permute(0, 3, 1, 2)  # 인덱스 번호로 바꾸어주는 것 # 이미지 개수, 채널 수, 이미지 너비, 높이
        print('shape of x_data : ', np.shape(self.x_data))
        self.y_data = torch.FloatTensor(y_data)  # float tensor / long tensor 로 숫자 속성을 정해줄 수 있음
        self.len = self.y_data.shape[0]

    # x,y를 튜플형태로 내보내기
    def __getitem__(self, index):
        # self.x_data = torch.reshape(self.x_data, [1, 1, len(self.x_data)])
        x = self.x_data[index].to('cuda')
        y = self.y_data[index].to('cuda')
        return x, y

    def __len__(self):
        return self.len
