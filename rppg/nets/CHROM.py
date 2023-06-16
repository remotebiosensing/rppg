import torch
import numpy as np

class CHROM(torch.nn.Module):
    def __init__(self):
        super(CHROM, self).__init__()
        
    def forward(self, batch_x):
        batch_x = torch.mean(batch_x, dim=(3, 4))
        batch_x = torch.permute(batch_x, (0, 2, 1))  # (B, N, C)
        batch_size, N, num_features = batch_x.shape
        bvp = []
        for b in range(batch_size):
            X = batch_x[b]
            Xcomp = 3*X[:, 0] - 2*X[:, 1]
            Ycomp = (1.5*X[:, 0])+X[:, 1]-(1.5*X[:, 2])
            sX = torch.std(Xcomp)
            sY = torch.std(Ycomp)
            alpha = sX/sY
            bvp.append( Xcomp - Ycomp * alpha)
        bvp = torch.cat(bvp)
        bvp = bvp.view(batch_size,-1)
        return bvp
