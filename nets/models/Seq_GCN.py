import numpy as np
import torch
import torch.nn as nn
from nets.models.gcn_utils import KNN_dist, View_selector, LocalGCN, NonLocalMP
from nets.models import PhysNet
class Seq_GCN(nn.Module):
    def __init__(self, fps = 30):
        super().__init__()
        self.fps = fps
        self.net = PhysNet.encoder_block()

        self.flatten = nn.Flatten()

        self.vertices = torch.tensor(self.fps).cuda()

        self.LocalGCN1 = LocalGCN(k=5,n_views=self.fps)
        self.NonLocalMP1 = NonLocalMP(n_view=self.fps)
        self.LocalGCN2 = LocalGCN(k=5, n_views=self.fps)
        self.NonLocalMP2 = NonLocalMP(n_view=self.fps)
        self.LocalGCN3 = LocalGCN(k=5, n_views=self.fps)

        self.View_selector1 = View_selector(n_views=self.fps, sampled_view=self.fps//2)
        self.View_selector2 = View_selector(n_views=self.fps//2, sampled_view=self.fps //4)

        self.cls = nn.Sequential(
            nn.Linear(5184,256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,1)
        )
    def forward(self,x):
        y = self.net(x)
        # y = self.flatten(y)
        vertices = self.vertices.unsqueeze(0).repeat(y.shape[1],1,1)

        y = self.LocalGCN1(y,vertices)
        y = self.NonLocalMP1(y)
        y = self.cls(y)
        return y

