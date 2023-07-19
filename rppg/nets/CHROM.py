import torch
import numpy as np
from scipy import signal

def overlap_frames(data, window_size, overlap_ratio):
    batch_size, seq_len, channels = data.size()
    new_seq_len = int((batch_size*seq_len)//(seq_len*overlap_ratio) -1)

    overlapped_data = torch.zeros(new_seq_len, window_size, channels, dtype=data.dtype, device=data.device)

    data = data.view(-1,3)

    start_idx = 0
    overlapped_window = int(seq_len*overlap_ratio)
    for i in range(new_seq_len):
        end_idx = start_idx + seq_len
        overlapped_data[i] = data[start_idx:end_idx]
        start_idx += overlapped_window

    return overlapped_data

class CHROM(torch.nn.Module):
    def __init__(self):
        super(CHROM, self).__init__()
        self.LPF = 0.7
        self.HPF = 2.5
        self.overlap = 0.5
        self.WinSec = 1.6
        FS = 30
        niq = 1.2* FS

        self.B,self.A = signal.butter(3, [self.LPF/niq,self.HPF/niq], 'bandpass')
        self.interval = int((FS*self.WinSec) *self.overlap)
    def forward(self, batch_x):

        batch_x = torch.permute(batch_x,(0,2,1,3,4))

        # batch_x = torch.permute(batch_x,(0,1,3,4,2))

        batch_x = torch.mean(batch_x, dim=[3,4])

        batch_size_org, window,_ = batch_x.shape
        bvp = torch.zeros(batch_size_org*window)
        batch_x = overlap_frames(batch_x,window,self.overlap)

        RGBBase = torch.mean(batch_x, axis=1, keepdim=True)
        RGBNorm = torch.true_divide(batch_x,RGBBase)

        #BGR

        batch_size_overlapped, N, num_features = batch_x.shape
        self.m = signal.windows.hann(N)
        for b in range(batch_size_overlapped):
            X = RGBNorm[b]
            Xcomp = 3*X[:, 0] - 2*X[:, 1]
            Ycomp = (1.5*X[:, 0])+X[:, 1]-(1.5*X[:, 1])
            Xcomp = Xcomp.cpu().detach().numpy()
            Xcomp = torch.from_numpy(
                signal.filtfilt(self.B, self.A, Xcomp, axis=0).copy())

            Ycomp = Ycomp.cpu().detach().numpy()
            Ycomp = torch.from_numpy(signal.filtfilt(self.B, self.A, Ycomp).copy())

            sX = torch.std(Xcomp)
            sY = torch.std(Ycomp)
            alpha = sX/sY
            Swin = Xcomp - Ycomp * alpha
            Swin = Swin * self.m
            # bvp.append(Swin)
            bvp[b*self.interval:(b+1)*self.interval] = bvp[b*self.interval:(b+1)*self.interval] + Swin[:self.interval]
            bvp[(b+1) * self.interval:(b + 2) * self.interval] = Swin[self.interval:]
        # bvp = torch.cat(bvp)
        bvp = bvp.view(batch_size_org,-1)
        return bvp
