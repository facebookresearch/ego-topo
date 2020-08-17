import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import kaiming_init

from ..registry import LFB_MODULES


@LFB_MODULES.register_module
class SimpleLfbModule(nn.Module):
    def __init__(self, pool_type='avg'):
        super(SimpleLfbModule, self).__init__()

        assert pool_type in ['avg', 'max']
        if pool_type == 'avg':
            self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.pool = nn.AdaptiveMaxPool3d((1, 1, 1))

    def init_weights(self):
        pass

    def forward(self, sfb, lfb):
        # (B, N, C) --> (B, C, N, 1, 1)
        nB, nN, nC = lfb.size()
        lfb = lfb.permute(0, 2, 1)
        lfb = lfb.reshape((nB, nC, nN, 1, 1))

        lfb = self.pool(lfb)

        return torch.cat((sfb, lfb), dim=1)
