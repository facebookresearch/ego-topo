import torch
import torch.nn as nn
import torch.nn.functional as F

from mmaction.models.registry import HEADS


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


@HEADS.register_module
class MultiClsHead(nn.Module):
    """Multi-task classification head"""

    def __init__(self,
                 with_avg_pool=True,
                 temporal_feature_size=1,
                 spatial_feature_size=7,
                 dropout_ratio=0.8,
                 in_channels=2048,
                 num_classes=[101],
                 init_std=0.01,
                 eval_topk=[1, 5],
                 return_feat=False):

        super(MultiClsHead, self).__init__()

        self.with_avg_pool = with_avg_pool
        self.dropout_ratio = dropout_ratio
        self.in_channels = in_channels
        self.dropout_ratio = dropout_ratio
        self.temporal_feature_size = temporal_feature_size
        self.spatial_feature_size = spatial_feature_size
        self.init_std = init_std
        self.num_task = len(num_classes)
        self.eval_topk = eval_topk
        self.return_feat = return_feat

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool3d(
                (temporal_feature_size, spatial_feature_size, spatial_feature_size))

        self.fc_cls = nn.ModuleList([nn.Linear(in_channels, n) for n in num_classes])

    def init_weights(self):
        for fc in self.fc_cls:
            nn.init.normal_(fc.weight, 0, self.init_std)
            nn.init.constant_(fc.bias, 0)

    def forward(self, x):
        if x.ndimension() == 4:
            x = x.unsqueeze(2)
        assert x.shape[1] == self.in_channels
        assert x.shape[2] == self.temporal_feature_size
        assert x.shape[3] == self.spatial_feature_size
        assert x.shape[4] == self.spatial_feature_size
        if self.with_avg_pool:
            x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)

        ret = [fc(x) for fc in self.fc_cls]
        if not self.training and self.return_feat:
            ret.append(x)
        return ret

    def loss(self,
             cls_score,
             labels):
        assert len(cls_score) == self.num_task
        if len(labels.shape) == 1:
            labels = labels.view(-1, 1)
        assert labels.shape[1] == self.num_task
        losses = dict()
        for i in range(self.num_task):
            losses['task_{}_loss_cls'.format(i)] = F.cross_entropy(cls_score[i], labels[:, i])
            topk = accuracy(cls_score[i], labels[:, i], topk=self.eval_topk)
            for k, result in zip(self.eval_topk, topk):
                losses['task_{}_top{}'.format(i, k)] = result

        return losses


@HEADS.register_module
class MultiClsHead1D(nn.Module):
    """Multi-task classification head"""

    def __init__(self,
                 in_channels=2048,
                 num_classes=[101],
                 dropout_ratio=0.0,
                 init_std=0.01,
                 eval_topk=[1, 5],
                 return_feat=False):

        super(MultiClsHead1D, self).__init__()

        self.in_channels = in_channels
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.num_task = len(num_classes)
        self.eval_topk = eval_topk
        self.return_feat = return_feat

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        self.fc_cls = nn.ModuleList([nn.Linear(in_channels, n) for n in num_classes])

    def init_weights(self):
        for fc in self.fc_cls:
            nn.init.normal_(fc.weight, 0, self.init_std)
            nn.init.constant_(fc.bias, 0)

    def forward(self, x):
        assert len(x.shape) == 2
        if self.dropout is not None:
            x = self.dropout(x)
    
        ret = [fc(x) for fc in self.fc_cls]
        if not self.training and self.return_feat:
            ret.append(x)
        return ret

    def loss(self,
             cls_score,
             labels):
        assert len(cls_score) == self.num_task
        if len(labels.shape) == 1:
            labels = labels.view(-1, 1)
        assert labels.shape[1] == self.num_task
        losses = dict()
        for i in range(self.num_task):
            mask = labels[:, i] >= 0
            losses['task_{}_loss_cls'.format(i)] = F.cross_entropy(cls_score[i][mask], labels[:, i][mask])
            topk = accuracy(cls_score[i][mask], labels[:, i][mask], topk=self.eval_topk)
            for k, result in zip(self.eval_topk, topk):
                losses['task_{}_top{}'.format(i, k)] = result

        return losses