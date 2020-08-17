import torch
import torch.nn as nn
import torch.nn.functional as F

from mmaction.models.registry import HEADS


@HEADS.register_module
class AffordanceHead(nn.Module):
    """Regression head"""

    def __init__(self, in_channels=2048, init_std=0.01, return_feat=False):
        super().__init__()
        self.init_std = init_std
        self.vdist = nn.Linear(in_channels, 125)
        self.ndist = nn.Linear(in_channels, 352)
        self.idist = nn.Linear(in_channels, 250)
        self.return_feat = return_feat

    def init_weights(self):
        for fc in [self.vdist, self.ndist, self.idist]:
            nn.init.normal_(fc.weight, 0, self.init_std)
            nn.init.constant_(fc.bias, 0)

    def forward(self, x, kwargs=None):

        x = x.view(x.shape[0], -1)

        if self.return_feat:
            return [self.vdist(x), x]

        vdist = self.vdist(x)
        vlabel = (kwargs['verb_dist']>0).float()
        vloss = F.binary_cross_entropy(torch.sigmoid(vdist), vlabel, reduction='none')
        vloss = (vloss*kwargs['vmask']).sum(1)/kwargs['vmask'].sum(1)

        ndist = self.ndist(x)
        nlabel = (kwargs['noun_dist']>0).float()
        nloss = F.binary_cross_entropy(torch.sigmoid(ndist), nlabel, reduction='none')
        nloss = (nloss*kwargs['nmask']).sum(1)/kwargs['nmask'].sum(1)

        return {'vdist_loss':vloss.mean(), 'ndist_loss':nloss.mean()}