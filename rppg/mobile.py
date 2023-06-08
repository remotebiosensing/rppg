import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter
import scipy
import json


# -------------------------------------------------------------------------------------------------------------------
# PhysNet model
# 
# the output is an ST-rPPG block rather than a rPPG signal.
# -------------------------------------------------------------------------------------------------------------------
def BPF(input_val, fs=30, low=0.75, high=2.5):
    low = low / (0.5 * fs)
    high = high / (0.5 * fs)
    [b_pulse, a_pulse] = butter(6, [low, high], btype='bandpass')
    return scipy.signal.filtfilt(b_pulse, a_pulse, np.double(input_val))


def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def calculate(signal, fs=60, low_pass=0.75, high_pass=2.5):
    signal = np.expand_dims(signal, 0)
    N = _next_power_of_2(signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(signal, fs=fs, nfft=N, detrend=False)
    fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
    mask_ppg = np.take(f_ppg, fmask_ppg)
    mask_pxx = np.take(pxx_ppg, fmask_ppg)
    data = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
    return data


class PhysNet(nn.Module):
    def __init__(self, S=2, in_ch=3):
        super().__init__()

        self.S = S  # S is the spatial dimension of ST-rPPG block

        self.start = nn.Sequential(
            nn.Conv3d(in_channels=in_ch, out_channels=32, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ELU()
        )

        # 1x
        self.loop1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )

        # encoder
        self.encoder1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.encoder2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )

        #
        self.loop4 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )

        # decoder to reach back initial temporal length
        self.decoder1 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )

        self.end = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, S, S)),
            nn.Conv3d(in_channels=64, out_channels=1, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))
        )

    def forward(self, x):
        means = torch.mean(x, dim=(2, 3, 4), keepdim=True)
        stds = torch.std(x, dim=(2, 3, 4), keepdim=True)
        x = (x - means) / stds  # (B, C, T, 128, 128)

        parity = []
        x = self.start(x)  # (B, C, T, 128, 128)
        x = self.loop1(x)  # (B, 64, T, 64, 64)
        parity.append(x.size(2) % 2)
        x = self.encoder1(x)  # (B, 64, T/2, 32, 32)
        parity.append(x.size(2) % 2)
        x = self.encoder2(x)  # (B, 64, T/4, 16, 16)
        x = self.loop4(x)  # (B, 64, T/4, 8, 8)

        x = F.interpolate(x, scale_factor=(2, 1, 1))  # (B, 64, T/2, 8, 8)
        x = self.decoder1(x)  # (B, 64, T/2, 8, 8)
        x = F.pad(x, (0, 0, 0, 0, 0, parity[-1]), mode='replicate')
        x = F.interpolate(x, scale_factor=(2, 1, 1))  # (B, 64, T, 8, 8)
        x = self.decoder2(x)  # (B, 64, T, 8, 8)
        x = F.pad(x, (0, 0, 0, 0, 0, parity[-2]), mode='replicate')
        # x = F.interpolate(x, scale_factor=(1,.25,.25))
        x = self.end(x)  # (B, 1, T, S, S), ST-rPPG block

        x_list = []
        for a in range(self.S):
            for b in range(self.S):
                x_list.append(x[:, :, :, a, b])  # (B, 1, T)

        x = sum(x_list) / (self.S * self.S)  # (B, 1, T)
        X = torch.cat(x_list + [x], 1)  # (B, M, T), flatten all spatial signals to the second dimension
        return X
class InterpolateLayer1(nn.Module):
    def __init__(self):
        super(InterpolateLayer1, self).__init__()

    def forward(self, x):
        x = F.interpolate(x, scale_factor=[2.0, 1.0, 1.0])
        return x
class InterpolateLayer2(nn.Module):
    def __init__(self):
        super(InterpolateLayer2, self).__init__()

    def forward(self, x):
        x = F.pad(x, (0, 0, 0, 0, 0, 1), mode='replicate')
        x = F.interpolate(x, scale_factor=[2.0, 1.0, 1.0])  # (B, 64, T, 8, 8)
        return x
class InterpolateLayer3(nn.Module):
    def __init__(self):
        super(InterpolateLayer3, self).__init__()

    def forward(self, x):
        x = F.pad(x, (0, 0, 0, 0, 0, 0), mode='replicate')
        x = F.interpolate(x, scale_factor=[1.0, 1 / 4, 1 / 4])  # (B, 64, T, 8, 8)
        return x

class SS(nn.Module):
    def __init__(self):
        super(SS, self).__init__()
    def forward(self,x):
        x_list = []
        for a in range(2):
            for b in range(2):
                x_list.append(x[:, :, :, a, b])  # (B, 1, T)
        x_tensor = torch.stack(x_list, dim=0)
        x_sum = torch.sum(x_tensor,dim=0)

        x = x_sum / (2 * 2)  # (B, 1, T)
        X = torch.cat(x_list + [x], 1)  # (B, M, T), flatten all spatial signals to the second dimension
        return X


if __name__ == "__main__":
    model = PhysNet().eval()
    model.load_state_dict(torch.load('./pretrain_model/model.pt'))
    model.eval()

    model_before_division = []
    model_after_division = []



    flag = True
    cnt = 0

    for name, module in model.named_modules():
        print(name, module)
        if name == '':
            continue
        if name == 'loop1':
            flag = False
        cnt +=1
        if name == 'start':
            start = module
        if name == 'loop1':
            loop1 = module
        if name == 'encoder1':
            encoder1 = module
        if name == 'encoder2':
            encoder2 = module
        if name == 'loop4':
            loop4 = module
        if name == 'decoder1':
            decoder1 = module
        if name == 'decoder2':
            decoder2 = module
        if name == 'end.1':
            end1 = module
        #
        # if flag:
        #     model_before_division.append(module)
        #     # test = module
        #     break
        # else:
        #     model_after_division.append(module)


    division_point ='end.0'

    model_before_division = torch.nn.Sequential(*model_before_division)
    model_after_division = torch.nn.Sequential(*model_after_division)

    # model_cnn = model_after_division[0][1]
    #

    i1 = InterpolateLayer1()
    i2 = InterpolateLayer2()
    i3 = InterpolateLayer3()
    ss = SS()

    combined_model = nn.Sequential(
        start,
        loop1,
        encoder1,
        encoder2,
        loop4,
        i1,
        decoder1,
        i2,
        decoder2,
        i3,
        end1,
        ss
    )

    scripted_model = torch.jit.script(combined_model)
    scripted_model.save('./pretrain_model/m_model')
    # video = sys.argv[1]
    dummy = np.zeros(shape=(1, 3, 30, 128, 128))
    out = combined_model(torch.Tensor(dummy))
    # out = model(torch.Tensor(dummy))
    # out = start(torch.Tensor(dummy))
    # out = loop1(out)
    # out = encoder1(out)
    # out = encoder2(out)
    # out = loop4(out)
    # out = F.interpolate(out, scale_factor=(2, 1, 1))
    # out = decoder1(out)
    # out = F.pad(out, (0, 0, 0, 0, 0, 1), mode='replicate')
    # out = F.interpolate(out, scale_factor=(2, 1, 1))  # (B, 64, T, 8, 8)
    # out = decoder2(out)
    # out = F.pad(out, (0, 0, 0, 0, 0, 0), mode='replicate')
    # out = F.interpolate(out, scale_factor=(1, 1/4, 1/4))  # (B, 64, T, 8, 8)
    # out = end1(out)
    # out = ss(out)
    rppg = out[:, -1, :][0]
