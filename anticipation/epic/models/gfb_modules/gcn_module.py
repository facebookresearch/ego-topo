import dgl
import dgl.function as fn
from dgl.nn.pytorch import GraphConv, GATConv
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import kaiming_init

from ..registry import GFB_MODULES


@GFB_MODULES.register_module
class GCN(nn.Module):
    def __init__(self, in_dim, h_dim, num_layers, dropout=0):
        super().__init__()
        x = in_dim
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(
                GraphConv(x, h_dim, activation=F.relu)
            )
            x = h_dim
        self.dropout = nn.Dropout(p=dropout)
    
    def init_weights(self):
        pass

    def forward(self, inputs, g):
        h = inputs
        for layer in self.layers:
            h = layer(g, h)
            h = self.dropout(h)
        return h



@GFB_MODULES.register_module
class GAT(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 h_dim,
                 heads,
                 activation=F.relu,
                 feat_drop=0,
                 attn_drop=0,
                 negative_slope=0.2,
                 residual=False):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, h_dim, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                h_dim * heads[l-1], h_dim, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))

    def forward(self, inputs, g):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h)
            if l < self.num_layers - 1:
                h = h.flatten(1)
            else:
                h = h.mean(1)

        return h


@GFB_MODULES.register_module
class GAT1(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 h_dim,
                 heads,
                 activation=F.relu,
                 feat_drop=0,
                 attn_drop=0,
                 negative_slope=0.2,
                 residual=False):
        super(GAT1, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, h_dim, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                h_dim * heads[l-1], h_dim, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))

    def forward(self, inputs, g):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h)
            # if l < self.num_layers - 1:
            h = h.flatten(1)
            # else:
            #     h = h.mean(1)

        return h