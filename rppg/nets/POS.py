import torch
from utils.funcs import detrend
from scipy import signal
import numpy as np

class POS(torch.nn.Module):
    def __init__(self):
        super(POS, self).__init__()
        self.fs = 30
        self.WinSec = 1.6


    def forward(self,x):

        x = torch.permute(x, (0, 2, 1, 3,4))  # (B, N, C)
        x = torch.mean(x, dim=(3, 4))

        batch_size, N, num_features = x.shape
        H = torch.zeros(batch_size, 1, N).to("cuda")

        for b in range(batch_size):
            RGB = x[b]  # Assume RGB preprocessing already done
            N = RGB.shape[0]
            l = int(self.fs * self.WinSec)#math.ceil(WinSec * fs)

            for n in range(N):
                m = n - l
                if m >= 0:
                    Cn = RGB[m:n, :] / torch.mean(RGB[m:n, :], dim=0)
                    Cn = torch.transpose(Cn, 0, 1)
                    S = torch.matmul(torch.tensor([[0, 1, -1], [-2, 1, 1]], dtype=torch.float).to("cuda"), Cn)
                    h = S[0, :] + (torch.std(S[0, :]) / torch.std(S[1, :])) * S[1, :]
                    mean_h = torch.mean(h)
                    h = h - mean_h
                    H[b, 0, m:n] = H[b, 0, m:n] + h

        BVP = H.cpu().numpy()
        BVP = BVP.squeeze()
        b, a = signal.butter(1, [0.75 / self.fs * 2, 3 / self.fs * 2], btype='bandpass')
        for i in range(len(BVP)):
            BVP[i] = detrend(BVP[i], 100)
            BVP[i] = signal.filtfilt(b, a, BVP[i].astype(np.double))
        BVP = torch.from_numpy(BVP.copy()).view(batch_size, -1)
        return BVP