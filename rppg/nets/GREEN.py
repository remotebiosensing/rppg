import torch

class GREEN(torch.nn.Module):
    def __init__(self):
        super(GREEN, self).__init__()
    def forward(self,x):
        # x[:,2] : R channel
        # x = x[:,1] # B T H W
        # B, T, H, W  = x.shape
        x = torch.mean(x, dim=(3,4))
        return x[:,1,:].reshape(-1)