import torch
import numpy as np

class LGI(torch.nn.Module):
    def __init__(self):
        super(LGI, self).__init__()

    def forward(self,batch_x):
        batch_x = torch.mean(batch_x, dim=(3, 4))
        # batch_x = torch.permute(batch_x, (0, 2, 1))  # (B, N, C)
        batch_size, num_features, N  = batch_x.shape
        batch_x = batch_x.view(batch_size,-1,num_features,N)


        bvp = []
        for b in range(batch_size):
            X = batch_x[b]  # 변환된 부분: 입력 신호를 PyTorch 텐서로 변환
            U, _, _ = torch.svd(X)  # 변환된 부분: numpy.linalg.svd 대신 torch.svd 사용
            S = U[:, :, 0]
            S = torch.unsqueeze(S, 2)  # 변환된 부분: np.expand_dims 대신 torch.unsqueeze 사용
            sst = torch.matmul(S, torch.transpose(S, 1, 2))  # 변환된 부분: np.swapaxes 대신 torch.transpose 사용
            p = torch.tile(torch.eye(3), (S.shape[0], 1, 1)).to("cuda") # 변환된 부분: np.tile 대신 torch.tile 사용
            P = p - sst
            Y = torch.matmul(P, X)
            bvp.append(Y[:, 1, :])
        bvp = torch.cat(bvp)
        bvp = bvp.view(batch_size,-1)

        return bvp


