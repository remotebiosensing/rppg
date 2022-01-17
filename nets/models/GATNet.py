import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nets.layers.GNNLayers import GATLayerAdj,GATLayerMultiHead,GATLayerEdgeAverage,GATLayerEdgeSoftmax

class GAT(nn.Module):
    def __init__(self, num_features, num_classes, num_heads=[2, 2, 2]):
        super(GAT, self).__init__()

        self.layer_heads = [1] + num_heads
        self.GAT_layer_sizes = [num_features, 32, 64, 64]

        self.MLP_layer_sizes = [self.layer_heads[-1] * self.GAT_layer_sizes[-1], 32, num_classes]
        self.MLP_acts = [F.relu, lambda x: x]

        self.GAT_layers = nn.ModuleList(
            [
                GATLayerMultiHead(d_in * heads_in, d_out, heads_out)
                for d_in, d_out, heads_in, heads_out in zip(
                self.GAT_layer_sizes[:-1],
                self.GAT_layer_sizes[1:],
                self.layer_heads[:-1],
                self.layer_heads[1:],
            )
            ]
        )
        self.MLP_layers = nn.ModuleList(
            [
                nn.Linear(d_in, d_out)
                for d_in, d_out in zip(self.MLP_layer_sizes[:-1], self.MLP_layer_sizes[1:])
            ]
        )

    def forward(self, x, adj, src, tgt, Msrc, Mtgt, Mgraph):
        for l in self.GAT_layers:
            x = l(x, adj, src, tgt, Msrc, Mtgt)
        x = torch.mm(Mgraph.t(), x)
        for layer, act in zip(self.MLP_layers, self.MLP_acts):
            x = act(layer(x))
        return x