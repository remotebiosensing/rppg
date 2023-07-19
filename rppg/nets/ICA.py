import torch
from rppg.utils.funcs import detrend
from scipy import linalg
from scipy import signal
import math
#TODO : need to modify
class ICA(torch.nn.Module):
    def __init__(self):
        super(ICA, self).__init__()

        self.LPF = 0.7
        self.HPF = 2.5
        fs = 30
        self.nf = 1/2 * fs


    def forward(self,x):
        x = torch.permute(x, (0,2,3,4,1)).cpu()
        batch, _, _, _, _ = x.shape
        x = torch.mean(x,dim=(2,3))

        BGRNorm = torch.zeros_like(x)

        for b in range(batch):
            for c in range(3):
                BGRDetrend = torch.Tensor(detrend(x[b,:,c],100))
                BGRNorm[b,:,c] = (BGRDetrend - torch.mean(BGRDetrend))/ torch.std(BGRDetrend)
        self.ica(BGRNorm.permute(0,2,1),3)


        return x

    def ica(self, X, Nsources, Wprev=0):
        nRows = X.shape[1]
        nCols = X.shape[2]
        if nRows > nCols:
            print(
                "Warning - The number of rows cannot be greater than the number of columns.")
            print("Please transpose input.")

        if Nsources > min(nRows, nCols):
            Nsources = min(nRows, nCols)
            print(
                'Warning - The number of sources cannot exceed the number of observation channels.')
            print('The number of sources will be reduced to the number of observation channels ', Nsources)

        Winv, Zhat = self.jade(X, Nsources, Wprev)
        W = torch.pinverse(Winv)
        return W, Zhat



    def jade(self, X, m, Wprev):
        def givens_rotation(a, b):
            r = torch.sqrt(a * a + b * b)
            c = a / r
            s = b / r
            return c, s

        batch_size = X.shape[0]
        T = X.shape[2]
        nem = m
        seuil = 1 / math.sqrt(T) / 100
        if m < X.shape[1]:
            D, U = torch.linalg.eigh(torch.matmul(X, X.transpose(1, 2)) / T)
            Diag = D[:, -1]
            k = torch.argsort(Diag)
            pu = Diag[k]
            ibl = torch.sqrt(
                pu[:, X.shape[1] - m:X.shape[1]] - torch.mean(pu[:, 0:X.shape[1] - m], dim=1, keepdim=True))
            bl = torch.reciprocal(ibl)
            W = torch.matmul(torch.diag_embed(bl), U[:, :, k[X.shape[1] - m:X.shape[1]]].transpose(1, 2))
            IW = torch.matmul(U[:, :, k[X.shape[1] - m:X.shape[1]]], torch.diag_embed(ibl))
        else:
            IW = torch.sqrt(torch.matmul(X, torch.transpose(X, 1, 2)) / T)
            W = torch.linalg.inv(IW)

        Y = torch.matmul(W, X)
        R = torch.matmul(Y, Y.transpose(1, 2)) / T
        C = torch.matmul(Y, Y.transpose(1, 2)) / T
        Q = torch.zeros((batch_size, m * m * m * m), device=X.device, dtype=X.dtype)
        index = 0

        for lx in range(m):
            Y1 = Y[:, lx, :]
            for kx in range(m):
                Yk1 = Y1 * torch.conj(Y[:, kx, :])
                for jx in range(m):
                    Yjk1 = Yk1 * torch.conj(Y[:, jx, :])
                    for ix in range(m):
                        Q[:, index] = torch.sum(Yjk1 / math.sqrt(T) * Y[:, ix, :].transpose(0, 1) / math.sqrt(T),
                                                dim=1) - R[:, ix, jx] * R[:, lx, kx] - R[:, ix, kx] * R[:, lx, jx] - C[
                                                                                                                     :,
                                                                                                                     ix,
                                                                                                                     lx] * torch.conj(
                            C[:, jx, kx])
                        index += 1

        # Compute and Reshape the significant Eigen
        D, U = torch.linalg.eigh(Q.view(batch_size, m * m, m * m))
        Diag = torch.abs(D[:, :, -1])
        K = torch.argsort(Diag, dim=-1)
        la = D.gather(dim=-1, index=K)
        M = torch.zeros((batch_size, m, nem * m), dtype=X.dtype, device=X.device)
        Z = torch.zeros((batch_size, m), dtype=X.dtype, device=X.device)
        h = m * m - 1
        for u in range(0, nem * m, m):
            Z = U[:, :, K[:, :, h]].view(batch_size, m, m)
            M[:, :, u:u + m] = la[:, :, h].unsqueeze(-1) * Z
            h = h - 1

        # Approximate the Diagonalization of the Eigen Matrices:
        B = torch.tensor([[1, 0, 0], [0, 1, 1], [0, 0, -1j, 0, 1j]], dtype=X.dtype, device=X.device)
        Bt = B.transpose(0, 1)

        encore = torch.ones(batch_size, dtype=torch.bool, device=X.device)
        if Wprev == 0:
            V = torch.eye(m, dtype=X.dtype, device=X.device).unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            V = torch.linalg.inv(Wprev)

        # Main Loop:
        while encore.any():
            encore = torch.zeros(batch_size, dtype=torch.bool, device=X.device)
            for p in range(m - 1):
                for q in range(p + 1, m):
                    Ip = torch.arange(p, nem * m, m)
                    Iq = torch.arange(q, nem * m, m)
                    g = torch.stack([M[:, p, Ip] - M[:, q, Iq], M[:, p, Iq], M[:, q, Ip]], dim=-1)
                    temp1 = torch.sum(g.unsqueeze(2) * g.unsqueeze(3), dim=-1)
                    temp2 = torch.matmul(B, temp1)
                    temp = torch.matmul(temp2, Bt)
                    D, vcp = torch.linalg.eigh(torch.real(temp))
                    K = torch.argsort(D, dim=-1)
                    la = D.gather(dim=-1, index=K)
                    angles = vcp.gather(dim=-1, index=K)[:, :, :, -1]
                    angles = torch.where(angles[:, 0, 0] < 0, -angles, angles)
                    c, s = givens_rotation(angles[:, 0, 0], angles[:, 1, 0])

                    mask = torch.abs(s) > seuil
                    if mask.any():
                        encore = encore | mask
                        pair = [p, q]
                        G = torch.stack([[c, -torch.conj(s)], [s, c]], dim=-1).unsqueeze(0)  # Givens Rotation
                        V[:, :, pair] = torch.matmul(V[:, :, pair], G)
                        M[:, pair, :] = torch.matmul(torch.conj(G.transpose(2, 3)), M[:, pair, :])
                        temp1 = c.unsqueeze(-1) * M[:, :, Ip] + s.unsqueeze(-1) * M[:, :, Iq]
                        temp2 = -torch.conj(s).unsqueeze(-1) * M[:, :, Ip] + c.unsqueeze(-1) * M[:, :, Iq]
                        temp = torch.cat((temp1, temp2), dim=-1)
                        M[:, :, Ip] = temp1
                        M[:, :, Iq] = temp2

        # Whiten the Matrix
        # Estimation of the Mixing Matrix and Signal Separation
        A = torch.matmul(IW, V)
        S = torch.matmul(torch.conj(V.transpose(1, 2)), Y)
        return A, S
