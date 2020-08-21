from mmaction.models import builder
from mmaction.models.recognizers import BaseRecognizer
from mmaction.models.registry import RECOGNIZERS

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tmodels
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence, PackedSequence
import re
import numpy as np
from mmcv.cnn import constant_init, kaiming_init

from sklearn.metrics import precision_score, recall_score, f1_score,  average_precision_score, precision_recall_fscore_support, roc_auc_score
from .future_recognizer_misc import GFBModelGCN1
from .future_recognizer import I3DModel, RNNModel
from .. import builder as epic_builder


@RECOGNIZERS.register_module
class GFBModelGCNExtractor(GFBModelGCN1):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("GFBModelGCNExtractor")

    def get_graph_feat(self, kwargs):
        bg = kwargs['gfb']
        bg.to(kwargs['labels'].device)

        # Average all features in each node
        feats, lengths = bg.ndata.pop('feats'), bg.ndata.pop('length')
        # print(feats.shape)
        feats = feats.sum(1)/lengths.unsqueeze(1).float()
        if self.pre_trans is not None:
            feats = self.pre_trans(feats)

        pre_feats = feats

        if self.gcn is not None:
            feats = self.gcn(feats, bg)
        bg.ndata['feats'] = feats
        gfb = dgl.mean_nodes(bg, 'feats')

        bg.ndata['pre_feats'] = pre_feats
        pfb = dgl.mean_nodes(bg, 'pre_feats')

        return pfb, gfb
    
    def forward_test(self, num_modalities, img_meta, **kwargs):
        pfb, gfb = self.get_graph_feat(kwargs)
        
        return pfb, gfb


@RECOGNIZERS.register_module
class GFBModelGCNExtractorNode(GFBModelGCN1):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("GFBModelGCNExtractorNode")

    def get_graph_feat(self, kwargs):
        bg = kwargs['gfb']
        bg.to(kwargs['labels'].device)

        # Average all features in each node
        feats, lengths = bg.ndata.pop('feats'), bg.ndata.pop('length')
        # print(feats.shape)
        feats = feats.sum(1)/lengths.unsqueeze(1).float()
        if self.pre_trans is not None:
            feats = self.pre_trans(feats)

        pre_feats = feats

        if self.gcn is not None:
            feats = self.gcn(feats, bg)
        # bg.ndata['feats'] = feats
        # gfb = dgl.mean_nodes(bg, 'feats')

        # bg.ndata['pre_feats'] = pre_feats
        # pfb = dgl.mean_nodes(bg, 'pre_feats')

        return pre_feats, feats
    
    def forward_test(self, num_modalities, img_meta, **kwargs):
        pfb, gfb = self.get_graph_feat(kwargs)
        
        return pfb, gfb


@RECOGNIZERS.register_module
class I3DModelExtractor(I3DModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward_test(self, num_modalities, img_meta, **kwargs):
        lfb = kwargs['lfb']
        lfb_mean = lfb.mean(1)
        x = self.backbone(lfb_mean)
        
        return lfb, x


@RECOGNIZERS.register_module
class RNNModelExtractor(RNNModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward_test(self, num_modalities, img_meta, **kwargs):
        lfb = kwargs['lfb']
        x, (ht, ct) = self.rnn(lfb)
       
        return lfb, x