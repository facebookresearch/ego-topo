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
from .future_recognizer import I3DModel, GFBModel
from .. import builder as epic_builder


@RECOGNIZERS.register_module
class GFBModel1(GFBModel):
    def __init__(self, num_classes, gfb_loss_weight=0.1, train_cfg=None, test_cfg=None):
        super().__init__(num_classes, gfb_loss_weight)
        self.node_fc = nn.Linear(2048, self.num_classes)

    def get_graph_feat(self, kwargs):

        losses = {}

        bg = kwargs['gfb']
        bg.to(kwargs['labels'].device)

        # Average all features in each node
        feats, lengths = bg.ndata.pop('feats'), bg.ndata.pop('length')
        # print(feats.shape, lengths.shape)
        feats = feats.sum(1)/lengths.unsqueeze(1).float()
        feats = self.backbone(feats)

        bg.ndata['feats'] = feats

        # Each node is a node level future prediction subproblem
        
        node_pred = self.node_fc(feats)
        node_pred = torch.sigmoid(node_pred)
        node_labels = bg.ndata.pop('labels')
        mask = node_labels.sum(1)>=0
        losses['gfb_loss'] = self.gfb_loss_weight*F.binary_cross_entropy(node_pred[mask], node_labels[mask])

        gfb = dgl.mean_nodes(bg, 'feats')

        return gfb, losses


@RECOGNIZERS.register_module
class GFBModelGCN(GFBModel):
    def __init__(self, gfb_module=None, **kwargs):
        super().__init__(**kwargs)
        print("GFBModelGCN")

        if gfb_module is not None:
            self.gcn = epic_builder.build_gfb_module(gfb_module)
        else:
            self.gcn = None

    def get_graph_feat(self, kwargs):

        losses = {}

        bg = kwargs['gfb']
        bg.to(kwargs['labels'].device)

        # Average all features in each node
        feats, lengths = bg.ndata.pop('feats'), bg.ndata.pop('length')
        feats = feats.sum(1)/lengths.unsqueeze(1).float()
        if self.gcn is not None:
            feats = self.gcn(feats, bg)
        bg.ndata['feats'] = feats

        # Each node is a node level future prediction subproblem
        feats = self.backbone(feats)
        node_pred = self.fc(feats)
        node_pred = torch.sigmoid(node_pred)
        node_labels = bg.ndata.pop('labels')
        mask = node_labels.sum(1)>=0
        losses['gfb_loss'] = self.gfb_loss_weight*F.binary_cross_entropy(node_pred[mask], node_labels[mask])

        gfb = dgl.mean_nodes(bg, 'feats')

        return gfb, losses


@RECOGNIZERS.register_module
class GFBModelGCN1(GFBModel):
    def __init__(self, gfb_module=None, pre_trans=None, **kwargs):
        super().__init__(**kwargs)
        print("GFBModelGCN1")

        if pre_trans:
            self.pre_trans = builder.build_backbone(pre_trans)
        else:
            self.pre_trans = None

        if gfb_module is not None:
            self.gcn = epic_builder.build_gfb_module(gfb_module)
        else:
            self.gcn = None

    def get_graph_feat(self, kwargs):

        losses = {}

        bg = kwargs['gfb']
        bg.to(kwargs['labels'].device)

        # Average all features in each node
        feats, lengths = bg.ndata.pop('feats'), bg.ndata.pop('length')
        # print(feats.shape)
        feats = feats.sum(1)/lengths.unsqueeze(1).float()
        if self.pre_trans is not None:
            feats = self.pre_trans(feats)
        if self.gcn is not None:
            feats = self.gcn(feats, bg)
        bg.ndata['feats'] = feats

        # Each node is a node level future prediction subproblem
        if self.backbone is not None:
            feats = self.backbone(feats)
        node_pred = self.fc(feats)
        node_pred = torch.sigmoid(node_pred)
        node_labels = bg.ndata.pop('labels')
        mask = node_labels.sum(1)>=0
        losses['gfb_loss'] = self.gfb_loss_weight*F.binary_cross_entropy(node_pred[mask], node_labels[mask])

        gfb = dgl.mean_nodes(bg, 'feats')

        return gfb, losses


@RECOGNIZERS.register_module
class GFBModelGCNResidual(GFBModel):
    def __init__(self, gfb_module=None, **kwargs):
        super().__init__(**kwargs)
        print("GFBModelGCNResidual")

        if gfb_module is not None:
            self.gcn = epic_builder.build_gfb_module(gfb_module)
            print(self.gcn)
        else:
            self.gcn = None

    def get_graph_feat(self, kwargs):

        losses = {}

        bg = kwargs['gfb']
        bg.to(kwargs['labels'].device)

        # Average all features in each node
        feats, lengths = bg.ndata.pop('feats'), bg.ndata.pop('length')
        feats = feats.sum(1)/lengths.unsqueeze(1).float()
        if self.gcn is not None:
            feats_gcn = self.gcn(feats, bg)
            feats = feats + feats_gcn
        bg.ndata['feats'] = feats

        # Each node is a node level future prediction subproblem
        feats = self.backbone(feats)
        node_pred = self.fc(feats)
        node_pred = torch.sigmoid(node_pred)
        node_labels = bg.ndata.pop('labels')
        mask = node_labels.sum(1)>=0
        losses['gfb_loss'] = self.gfb_loss_weight*F.binary_cross_entropy(node_pred[mask], node_labels[mask])

        gfb = dgl.mean_nodes(bg, 'feats')

        return gfb, losses


@RECOGNIZERS.register_module
class GFBModelGCN2(GFBModel):
    def __init__(self, gfb_module=None, **kwargs):
        super().__init__(**kwargs)
        print("GFBModelGCN2")

        if gfb_module is not None:
            self.gcn = epic_builder.build_gfb_module(gfb_module)
        else:
            self.gcn = None

    def get_graph_feat(self, kwargs):

        losses = {}

        bg = kwargs['gfb']
        bg.to(kwargs['labels'].device)

        # Average all features in each node
        feats, lengths = bg.ndata.pop('feats'), bg.ndata.pop('length')
        feats = feats.sum(1)/lengths.unsqueeze(1).float()
        feats = self.backbone(feats)
        if self.gcn is not None:
            feats = self.gcn(feats, bg)
        bg.ndata['feats'] = feats

        # Each node is a node level future prediction subproblem
        node_pred = self.fc(feats)
        node_pred = torch.sigmoid(node_pred)
        node_labels = bg.ndata.pop('labels')
        mask = node_labels.sum(1)>=0
        losses['gfb_loss'] = self.gfb_loss_weight*F.binary_cross_entropy(node_pred[mask], node_labels[mask])

        gfb = dgl.mean_nodes(bg, 'feats')

        return gfb, losses


@RECOGNIZERS.register_module
class GFBModelGCNTimeX(GFBModelGCN1):
    def __init__(self, timex=None, **kwargs):
        super().__init__(**kwargs)
        print("GFBModelGCNTimeX")

        if timex is not None:
            self.timex = builder.build_backbone(timex)
        else:
            self.timex = None

    def get_graph_feat(self, kwargs):

        losses = {}

        bg = kwargs['gfb']
        bg.to(kwargs['labels'].device)

        # Average all features in each node
        feats, lengths = bg.ndata.pop('feats'), bg.ndata.pop('length')

        if self.timex is not None:
            nB, nL, nC = feats.size()
            x = feats.permute(0, 2, 1)
            x = x.reshape((nB, nC, nL, 1, 1))
            # print(x.shape)
            x = self.timex(x) # nB, nC, nL / K, 1, 1
            # print(x.shape)
            feats = x.mean(2).reshape(nB, -1)
            # feats = self.timex(x)
        # print(feats.shape)
        # feats = feats.sum(1)/lengths.unsqueeze(1).float()
        if self.pre_trans is not None:
            feats = self.pre_trans(feats)
        if self.gcn is not None:
            feats = self.gcn(feats, bg)
        bg.ndata['feats'] = feats

        # Each node is a node level future prediction subproblem
        if self.backbone is not None:
            feats = self.backbone(feats)
        node_pred = self.fc(feats)
        node_pred = torch.sigmoid(node_pred)
        node_labels = bg.ndata.pop('labels')
        mask = node_labels.sum(1)>=0
        losses['gfb_loss'] = self.gfb_loss_weight*F.binary_cross_entropy(node_pred[mask], node_labels[mask])

        gfb = dgl.mean_nodes(bg, 'feats')

        return gfb, losses