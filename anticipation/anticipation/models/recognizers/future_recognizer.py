# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

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
from .. import builder as epic_builder

@RECOGNIZERS.register_module
class I3DModel(BaseRecognizer):
    def __init__(self, num_classes, backbone, fc_indim=2048, train_cfg=None, test_cfg=None):
        super().__init__()

        if backbone is not None:
            self.backbone = builder.build_backbone(backbone)
        else:
            self.backbone = None

        # if label=='int':
        #     self.num_classes = 250
        # elif label=='noun':
        #     self.num_classes = 352
        self.num_classes = num_classes
        self.fc_indim = fc_indim
        self.fc = nn.Linear(fc_indim, self.num_classes)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

      
    def parameters(self):
        output = filter(lambda p: p.requires_grad, super().parameters())
        return output

    def eval_mAP(self, pred, labels, label_mask):
        mAPs = dict()
        for i in range(label_mask.shape[1]):
            # print(label_mask[0][i])
            pred_i = pred[:, label_mask[0][i]]
            labels_i = labels[:, label_mask[0][i]]
            # print(pred_i.shape, pred.shape)
            mAPs['mAP_{}'.format(i)] = (pred_i.detach().cpu(), labels_i.detach().cpu())
        
        return mAPs
    
    def eval_ratio_mAP(self, pred, labels, label_mask, ratio_idx):
        mAPs = dict()
        m = torch.max(ratio_idx).item()
        for i in range(m + 1):
            # print(label_mask[0][i])
            #idx = ratio_idx == i
            # print(pred.shape, ratio_idx.shape)
            pred_i, labels_i = pred[ratio_idx[:, 0] == i], labels[ratio_idx[:, 0] == i]
            pred_i = pred_i[:, label_mask[0][0]]
            labels_i = labels_i[:, label_mask[0][0]]
            # print(pred_i.shape, pred.shape)
            if pred_i.shape[0] > 0:
                mAPs['mAP_ratio_{}'.format(i)] = (pred_i.detach().cpu(), labels_i.detach().cpu())
        
        return mAPs

    def forward_train(self, num_modalities, img_meta, gt_label, **kwargs): 
        losses = dict()

        lfb = kwargs['lfb'].mean(1) # (B, lfb_win=64, 2048) --> (B, 2048)
        x = self.backbone(lfb)
        pred = torch.sigmoid(self.fc(x))
        losses['cls_loss'] = F.binary_cross_entropy(pred, kwargs['labels'])
        losses.update(self.eval_mAP(pred, kwargs['labels'], kwargs['label_mask']))
        losses.update(self.eval_ratio_mAP(pred, kwargs['labels'], kwargs['label_mask'], kwargs['ratio_idx']))

        return losses

    def forward_test(self, num_modalities, img_meta, **kwargs):
        lfb = kwargs['lfb'].mean(1)
        x = self.backbone(lfb)
        pred = self.fc(x)
        return torch.sigmoid(pred)


@RECOGNIZERS.register_module
class GFBModel(I3DModel):
    def __init__(self, gfb_loss_weight=0.1, new_node_fc=False, **kwargs):
        super().__init__(**kwargs)
        self.gfb_loss_weight = gfb_loss_weight
        self.new_node_fc = new_node_fc
        if new_node_fc:
            self.nfc = nn.Linear(self.fc_indim, self.num_classes)

    def get_graph_feat(self, kwargs):

        losses = {}

        bg = kwargs['gfb']
        bg.to(kwargs['labels'].device)

        # Average all features in each node
        feats, lengths = bg.ndata.pop('feats'), bg.ndata.pop('length')
        feats = feats.sum(1)/lengths.unsqueeze(1).float()
        bg.ndata['feats'] = feats

        # Each node is a node level future prediction subproblem
        if self.backbone is not None:
            feats = self.backbone(feats)
        if self.new_node_fc:
            node_pred = self.nfc(feats)
        else:
            node_pred = self.fc(feats)
        node_pred = torch.sigmoid(node_pred)
        node_labels = bg.ndata.pop('labels')
        mask = node_labels.sum(1)>=0
        losses['gfb_loss'] = self.gfb_loss_weight*F.binary_cross_entropy(node_pred[mask], node_labels[mask])

        gfb = dgl.mean_nodes(bg, 'feats')

        return gfb, losses


    def forward_train(self, num_modalities, img_meta, gt_label, **kwargs): 
        losses = dict()

        gfb, g_losses = self.get_graph_feat(kwargs)
        losses.update(g_losses)

        if self.backbone is not None:
            gfb = self.backbone(gfb)
        pred = torch.sigmoid(self.fc(gfb))
        losses['cls_loss'] = F.binary_cross_entropy(pred, kwargs['labels'])

        losses.update(self.eval_mAP(pred, kwargs['labels'], kwargs['label_mask']))
        losses.update(self.eval_ratio_mAP(pred, kwargs['labels'], kwargs['label_mask'], kwargs['ratio_idx']))


        return losses

    def forward_test(self, num_modalities, img_meta, **kwargs):
        gfb, _ = self.get_graph_feat(kwargs)
        if self.backbone is not None:
            gfb = self.backbone(gfb)
        pred = self.fc(gfb)
        return torch.sigmoid(pred)
    

@RECOGNIZERS.register_module
class RNNModel(I3DModel):
    def __init__(self, lstm, **kwargs):
        super().__init__(**kwargs)

        self.rnn = builder.build_backbone(lstm)

    def forward_train(self, num_modalities, img_meta, gt_label, **kwargs): 
        losses = dict()

        lfb = kwargs['lfb'] # (B, lfb_win=64, 2048)
        x, (ht, ct) = self.rnn(lfb) # (B, lfb_win=64, 2048)
        x = x.mean(1) # (B, lfb_win=64, 2048) --> (B, 2048)
        x = self.backbone(x)
        # x = x[:, -1, :]
        pred = torch.sigmoid(self.fc(x))
        losses['cls_loss'] = F.binary_cross_entropy(pred, kwargs['labels'])
        losses.update(self.eval_mAP(pred, kwargs['labels'], kwargs['label_mask']))
        losses.update(self.eval_ratio_mAP(pred, kwargs['labels'], kwargs['label_mask'], kwargs['ratio_idx']))


        return losses

    def forward_test(self, num_modalities, img_meta, **kwargs):
        lfb = kwargs['lfb']
        x, (ht, ct) = self.rnn(lfb)
        x = x.mean(1)
        x = self.backbone(x)
        # x = x[:, -1, :]
        pred = self.fc(x)
        return torch.sigmoid(pred)


@RECOGNIZERS.register_module
class TimeXModel(I3DModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward_train(self, num_modalities, img_meta, gt_label, **kwargs): 
        losses = dict()

        lfb = kwargs['lfb'] # (B, lfb_win=64, 2048)
        nB, nL, nC = lfb.size()
        x = lfb.permute(0, 2, 1)
        x = x.reshape((nB, nC, nL, 1, 1))
        # print(x.shape)
        x = self.backbone(x) # nB, nC, nL / K, 1, 1
        # print(x.shape)
        x = x.mean(2).reshape(nB, -1)
        # print(x.shape)
        pred = torch.sigmoid(self.fc(x))
        losses['cls_loss'] = F.binary_cross_entropy(pred, kwargs['labels'])
        losses.update(self.eval_mAP(pred, kwargs['labels'], kwargs['label_mask']))
        losses.update(self.eval_ratio_mAP(pred, kwargs['labels'], kwargs['label_mask'], kwargs['ratio_idx']))


        return losses

    def forward_test(self, num_modalities, img_meta, **kwargs):
        lfb = kwargs['lfb']
        nB, nL, nC = lfb.size()
        x = lfb.permute(0, 2, 1)
        x = x.reshape((nB, nC, nL, 1, 1))
        x = self.backbone(x) # nB, nC, nL / K, 1, 1
        x = x.mean(2).reshape(nB, -1)
        pred = torch.sigmoid(self.fc(x))
        return pred


@RECOGNIZERS.register_module
class NetVladModel(I3DModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward_train(self, num_modalities, img_meta, gt_label, **kwargs): 
        losses = dict()

        lfb = kwargs['lfb'] # (B, lfb_win=64, 2048)
        # lfb = self.reduce_fc(lfb)
        nB, nL, nC = lfb.size()
        x = lfb.permute(0, 2, 1)
        x = x.reshape((nB, nC, nL, 1))
        x = self.backbone(x) # nB, nC, nL / K, nL, 1
        # print(x.shape)
        pred = torch.sigmoid(self.fc(x))
        losses['cls_loss'] = F.binary_cross_entropy(pred, kwargs['labels'])
        losses.update(self.eval_mAP(pred, kwargs['labels'], kwargs['label_mask']))
        losses.update(self.eval_ratio_mAP(pred, kwargs['labels'], kwargs['label_mask'], kwargs['ratio_idx']))


        return losses

    def forward_test(self, num_modalities, img_meta, **kwargs):
        lfb = kwargs['lfb']
        nB, nL, nC = lfb.size()
        x = lfb.permute(0, 2, 1)
        x = x.reshape((nB, nC, nL, 1, 1))
        x = self.backbone(x) # nB, nC, nL / K, 1, 1
        # x = x.mean(2).reshape(nB, -1)
        pred = torch.sigmoid(self.fc(x))
        return pred


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