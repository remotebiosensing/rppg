import torch
import numpy as np
class PBV(torch.nn.Module):
    def __init__(self):
        super(PBV, self).__init__()

    def forward(self,batch_x):
        batch_x = torch.mean(batch_x, dim=(3, 4))
        # batch_x = torch.permute(batch_x, (0, 2, 1))  # (B, N, C)
        batch_size, num_features, N  = batch_x.shape
        # batch_x = batch_x.view(batch_size,-1,num_features,N)

        sig_mean = torch.mean(batch_x, dim=2)

        signal_norm_r = batch_x[:, 0, :] / sig_mean[:, 0].unsqueeze(1)
        signal_norm_g = batch_x[:, 1, :] / sig_mean[:, 1].unsqueeze(1)
        signal_norm_b = batch_x[:, 2, :] / sig_mean[:, 2].unsqueeze(1)

        pbv_n = torch.stack([torch.std(signal_norm_r, dim=1),
                             torch.std(signal_norm_g, dim=1),
                             torch.std(signal_norm_b, dim=1)], dim=0)

        pbv_d = torch.sqrt(torch.var(signal_norm_r, dim=1) +
                           torch.var(signal_norm_g, dim=1) +
                           torch.var(signal_norm_b, dim=1))

        pbv = pbv_n / pbv_d

        C = torch.stack([signal_norm_r, signal_norm_g, signal_norm_b], dim=1)
        # C = torch.permute(C,(0,2,1))
        Ct = torch.transpose(C, 1, 2)
        Q = torch.matmul(C, Ct)
        W = torch.linalg.solve(Q, pbv.permute(1, 0))

        A = torch.matmul(Ct, W.unsqueeze(2))
        B = torch.matmul(pbv.T.unsqueeze(1), W.unsqueeze(2))
        bvp = A / B
        return bvp.squeeze(dim=2)