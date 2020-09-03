# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging

import torch.nn as nn
import torch.utils.checkpoint as cp

from mmaction.models.registry import BACKBONES
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint


@BACKBONES.register_module
class mlp(nn.Module):
    """mlp backbone.
    """
    def __init__(
        self,
        num_layers=2,
        in_channels=2048,
        h_channels=2048,
        out_channels=2048
    ):
        super(mlp, self).__init__()
        
        layers = []
        in_dim = in_channels
        for i in range(num_layers):
            out_dim = h_channels if i < num_layers - 1 else out_channels
            layers.append(
                nn.Conv3d(in_dim, out_dim, kernel_size=(1, 1, 1), bias=False)
            )
            layers.append(nn.ReLU(inplace=True))
            in_dim = out_dim
        
        self.mlp_layers = nn.Sequential(*layers)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                kaiming_init(m)

    def forward(self, x):
       return self.mlp_layers(x)


@BACKBONES.register_module
class mlp1D(nn.Module):
    """mlp 1D backbone.
    """
    def __init__(
        self,
        num_layers=2,
        in_channels=2048,
        h_channels=2048,
        out_channels=2048
    ):
        super(mlp1D, self).__init__()
        
        layers = []
        in_dim = in_channels
        for i in range(num_layers):
            out_dim = h_channels if i < num_layers - 1 else out_channels
            layers.append(
                nn.Linear(in_dim, out_dim)
            )
            layers.append(nn.ReLU(inplace=True))
            in_dim = out_dim
        
        self.mlp_layers = nn.Sequential(*layers)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                kaiming_init(m)

    def forward(self, x):
       return self.mlp_layers(x)

