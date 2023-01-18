import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class LSTMAutoEncoderDataset(Dataset):
    def __init__(self, input_signal, target_signal, mean_target_signal, std_target_signal):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.input_signal = input_signal
        self.target_signal = target_signal
        self.mean = mean_target_signal
        self.std = std_target_signal

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        input_signal = torch.tensor(self.input_signal[index], dtype=torch.float32).transpose(1, 0)

        if self.target_signal.ndim < 3:
            target_signal = torch.tensor(self.target_signal[index], dtype=torch.float32)
        else:
            target_signal = torch.tensor(self.target_signal[index], dtype=torch.float32).transpose(1, 0)

        if torch.cuda.is_available():
            input_signal = input_signal.to('cuda')
            target_signal = target_signal.to('cuda')

        return input_signal, target_signal

    def __len__(self):
        return len(self.input_signal)

    def get_mean_std(self):
        return self.mean, self.std
