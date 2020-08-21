import dgl
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from mmaction.models import builder
from mmaction.models.recognizers import BaseRecognizer
from mmaction.models.registry import RECOGNIZERS

from .. import builder as epic_builder


@RECOGNIZERS.register_module
class NodeLSTM(BaseRecognizer):
    def __init__(self,
                 backbone,
                 reg_head=None,
                 cls_head=None,
                 train_cfg=None,
                 test_cfg=None):

        super(NodeLSTM, self).__init__()

        self.backbone = builder.build_backbone(backbone)

        if reg_head is not None:
            self.reg_head = builder.build_head(reg_head)
        # else:
        #     print("model without head (may for feature extraction)")
        
        if cls_head is not None:
            self.cls_head = builder.build_head(cls_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights()

    @property
    def with_reg_head(self):
        return hasattr(self, 'reg_head') and self.reg_head is not None

    @property
    def with_cls_head(self):
        return hasattr(self, 'cls_head') and self.cls_head is not None

    def init_weights(self):
        super(NodeLSTM, self).init_weights()

        self.backbone.init_weights()
        
        if self.with_reg_head:
            self.reg_head.init_weights()
        
        if self.with_cls_head:
            self.cls_head.init_weights()

    def forward_train(self,
                      num_modalities,
                      img_meta,
                      **kwargs): 
        x = kwargs['feature']
        length = kwargs['length'].reshape(-1)
        targets = x[:, 1:]
        x = x[:, :-1]
        packed_x = pack_padded_sequence(x, length, batch_first=True, enforce_sorted=False)
        packed_targets = pack_padded_sequence(targets, length, batch_first=True, enforce_sorted=False)

        out, (ht, ct) = self.backbone(packed_x)
        losses = dict()
       
        # print(x.shape)

        if self.with_reg_head:
            x = self.reg_head(out.data)
            #print(x.shape, packed_targets.data.shape)
            loss_reg = self.reg_head.loss(x, packed_targets.data)
            losses.update(loss_reg)
        
        if self.with_cls_head:
            x = self.cls_head(out.data)
            label = kwargs['gt_label'][:, 1:]
            packed_labels = pack_padded_sequence(label, length, batch_first=True, enforce_sorted=False)
            loss_cls = self.cls_head.loss(x, packed_labels.data)
            losses.update(loss_cls)
        
        # x, _ = pad_packed_sequence(x, batch_first=True)
        # idx = packed_x.sorted_indices.sort(0)
        # x = x[idx]

        return losses

    def forward_test(self,
                     num_modalities,
                     img_meta,
                     **kwargs):
        x = kwargs['feature']
        assert x.shape[0] == 1
        length = kwargs['length'].reshape(-1)
        # targets = x[:, 1:]
        x = x[:, :-1]

        # print("before", x.shape)

        packed_x = pack_padded_sequence(x, length, batch_first=True, enforce_sorted=False)
        # packed_targets = pack_padded_sequence(targets, length, batch_first=True, enforce_sorted=False)

        # print(packed_x.data.shape)

        out, (ht, ct) = self.backbone(packed_x)
        # x = x.data

        x = ht[-1]

        if self.with_reg_head:
            x = self.reg_head(x)
        
        # x = x[length[0] - 1]

        ret = x.cpu().numpy()
        return ret
