import torch
import torch.nn as nn
import torch.nn.functional as F

from mmaction.models.registry import HEADS


@HEADS.register_module
class RegHead(nn.Module):
    """Regression head"""

    def __init__(self, in_channels=2048, out_channels=2048, init_std=0.01):
        super(RegHead, self).__init__()

        self.init_std = init_std

        self.fc = nn.Linear(in_channels, out_channels)

    def init_weights(self):
        nn.init.normal_(self.fc.weight, 0, self.init_std)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
       return self.fc(x)

    def loss(self, pred, target):
        losses = dict()
        losses['mse_loss'] = F.mse_loss(pred, target, reduction='none').mean(1)

        return losses