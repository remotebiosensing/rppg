import numpy as np
import torch
import torch.nn as nn
from nets.models.gcn_utils import KNN_dist, View_selector, LocalGCN, NonLocalMP
from nets.models import PhysNet
class Seq_GCN(nn.Module):
    def __init__(self, fps = 32):
        super().__init__()
        self.Local = []
        phi = (1 + np.sqrt(5)) / 2
        vertices = [[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
                    [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
                    [0, 1 / phi, phi], [0, 1 / phi, -phi], [0, -1 / phi, phi], [0, -1 / phi, -phi],
                    [phi, 0, 1 / phi], [phi, 0, -1 / phi], [-phi, 0, 1 / phi], [-phi, 0, -1 / phi],
                    [1 / phi, phi, 0], [-1 / phi, phi, 0], [1 / phi, -phi, 0], [-1 / phi, -phi, 0]]
        # for i in range(fps):
        #     vertices.append(i)
        self.vertices = torch.tensor(vertices).cuda()
        for i in range(128):
            self.Local.append(LocalModule().cuda())
        self.fps = fps
        self.net = PhysNet.encoder_block()

        self.flatten = nn.Flatten()



        # self.vertices = torch.tensor(self.fps).cuda()



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
        views = self.fps
        y = []
        [batch, channel, length, width, height] = x.shape
        x = torch.permute(x,[2,0,1,3,4])
        for i in range(length):
            y.append(self.Local[i](x[0],self.vertices))


        y = self.net(x)
        y = y.view((int(x.shape[0]/views),views,-1))
        vertices = self.vertices.unsqueeze(0).repeat(y.shape[0],1,1)
        y = self.LocalGCN1(y, vertices)
        y = self.flatten(y)

        return y.view(-1,32) # 32 length

class LocalModule(nn.Module):
    def __init__(self):
        super(LocalModule, self).__init__()
        self.conv = nn.Conv2d(3,1,3)
        self.LocalGCN = LocalGCN(k=5,n_views=1)
        # self.NonLocalMP = NonLocalMP(1)
    def forward(self,x,vertices):
        #임의의 view
        views = 32
        y = self.conv(x)
        y = y.view((int(x.shape[0] / views), views, -1))
        vertices = vertices.unsqueeze(0).repeat(y.shape[0], 1, 1)
        y = self.LocalGCN(y,vertices)
        # y = self.NonLocalMP(y)
        return y
