import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GATLayerAdj(nn.Module):
    """
    More didatic (also memory-hungry) GAT layer
    """

    def __init__(self, d_i, d_o, act=F.relu, eps=1e-6):
        super(GATLayerAdj, self).__init__()
        self.f = nn.Linear(2 * d_i, d_o)
        self.w = nn.Linear(2 * d_i, 1)
        self.act = act
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.f.weight)
        nn.init.xavier_uniform_(self.w.weight)

    def forward(self, x, adj, src, tgt, Msrc, Mtgt):
        """
        features -> N,i node features
        adj -> N,N adjacency matrix
        src -> E,i source index for edges
        tgt -> E,i target index for edges
        Msrc -> N,E adjacency matrix from source nodes to edges
        Mtgt -> N,E adjacency matrix from target nodes to edges
        """
        N = x.size()[0]
        hsrc = x.unsqueeze(0).expand(N, -1, -1)  # 1,N,i
        htgt = x.unsqueeze(1).expand(-1, N, -1)  # N,1,i

        h = torch.cat([hsrc, htgt], dim=2)  # N,N,2i

        a = self.w(h)  # N,N,1
        a_sqz = a.squeeze(2)  # N,N
        a_zro = -1e16 * torch.ones_like(a_sqz)  # N,N
        a_msk = torch.where(adj > 0, a_sqz, a_zro)  # N,N
        a_att = F.softmax(a_msk, dim=1)  # N,N

        y = self.act(self.f(h))  # N,N,o
        y_att = a_att.unsqueeze(-1) * y  # N,N,o
        o = y_att.sum(dim=1).squeeze()

        return o


class GATLayerEdgeAverage(nn.Module):
    """
    GAT layer with average, instead of softmax, attention distribution
    """

    def __init__(self, d_i, d_o, act=F.relu, eps=1e-6):
        super(GATLayerEdgeAverage, self).__init__()
        self.f = nn.Linear(2 * d_i, d_o)
        self.w = nn.Linear(2 * d_i, 1)
        self.act = act
        self._init_weights()
        self.eps = eps

    def _init_weights(self):
        nn.init.xavier_uniform_(self.f.weight)
        nn.init.xavier_uniform_(self.w.weight)

    def forward(self, x, adj, src, tgt, Msrc, Mtgt):
        """
        features -> N,i node features
        adj -> N,N adjacency matrix
        src -> E,i source index for edges
        tgt -> E,i target index for edges
        Msrc -> N,E adjacency matrix from source nodes to edges
        Mtgt -> N,E adjacency matrix from target nodes to edges
        """
        hsrc = x[src]  # E,i
        htgt = x[tgt]  # E,i
        h = torch.cat([hsrc, htgt], dim=1)  # E,2i
        y = self.act(self.f(h))  # E,o
        a = self.w(h)  # E,1
        a_sum = torch.mm(Mtgt, a) + self.eps  # N,E x E,1 = N,1
        o = torch.mm(Mtgt, y * a) / a_sum  # N,1
        assert not torch.isnan(o).any()

        return o


class GATLayerEdgeSoftmax(nn.Module):
    """
    GAT layer with softmax attention distribution (May be prone to numerical errors)
    """

    def __init__(self, d_i, d_o, act=F.relu, eps=1e-6):
        super(GATLayerEdgeSoftmax, self).__init__()
        self.f = nn.Linear(2 * d_i, d_o)
        self.w = nn.Linear(2 * d_i, 1)
        self.act = act
        self._init_weights()
        self.eps = eps

    def _init_weights(self):
        nn.init.xavier_uniform_(self.f.weight)
        nn.init.xavier_uniform_(self.w.weight)

    def forward(self, x, adj, src, tgt, Msrc, Mtgt):
        """
        features -> N,i node features
        adj -> N,N adjacency matrix
        src -> E,i source index for edges
        tgt -> E,i target index for edges
        Msrc -> N,E adjacency matrix from source nodes to edges
        Mtgt -> N,E adjacency matrix from target nodes to edges
        """
        hsrc = x[src]  # E,i
        htgt = x[tgt]  # E,i
        h = torch.cat([hsrc, htgt], dim=1)  # E,2i
        y = self.act(self.f(h))  # E,o
        # FIXME Manual softmax doesn't as expected numerically
        a = self.w(h)  # E,1
        assert not torch.isnan(a).any()
        a_base, _ = torch.max(a, 0, keepdim=True)  # [0] + self.eps
        assert not torch.isnan(a_base).any()
        a_norm = a - a_base
        assert not torch.isnan(a_norm).any()
        a_exp = torch.exp(a_norm)
        assert not torch.isnan(a_exp).any()
        a_sum = torch.mm(Mtgt, a_exp) + self.eps  # N,E x E,1 = N,1
        assert not torch.isnan(a_sum).any()
        o = torch.mm(Mtgt, y * a_exp) / a_sum  # N,1
        assert not torch.isnan(o).any()

        return o


class GATLayerMultiHead(nn.Module):

    def __init__(self, d_in, d_out, num_heads):
        super(GATLayerMultiHead, self).__init__()

        self.GAT_heads = nn.ModuleList(
            [
                GATLayerEdgeSoftmax(d_in, d_out)
                for _ in range(num_heads)
            ]
        )

    def forward(self, x, adj, src, tgt, Msrc, Mtgt):
        return torch.cat([l(x, adj, src, tgt, Msrc, Mtgt) for l in self.GAT_heads], dim=1)


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'