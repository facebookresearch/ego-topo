# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging

import torch.nn as nn
import torch.utils.checkpoint as cp

from mmaction.models.registry import BACKBONES
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint


@BACKBONES.register_module
class lstm(nn.Module):
    """lstm backbone.
    """
    def __init__(
        self,
        num_layers=2,
        in_channels=2048,
        out_channels=2048,
        dropout=0.0,
    ):
        super(lstm, self).__init__()
        
        self.lstm = nn.LSTM(in_channels, out_channels, num_layers=num_layers, dropout=dropout)

    def init_weights(self):
        pass

    def forward(self, x):

        out = self.lstm(x)

        return out
