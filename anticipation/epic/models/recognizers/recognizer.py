import dgl
import torch

from mmaction.models import builder
from mmaction.models.recognizers import BaseRecognizer
from mmaction.models.registry import RECOGNIZERS

from .. import builder as epic_builder


@RECOGNIZERS.register_module
class Recognizer(BaseRecognizer):
    def __init__(self,
                 backbone,
                 append_aux=False,
                 lfb_module=None,
                 gfb_module=None,
                 cls_head=None,
                 train_cfg=None,
                 test_cfg=None):

        super(Recognizer, self).__init__()

        self.backbone = builder.build_backbone(backbone)

        if lfb_module is not None:
            self.lfb_module = epic_builder.build_lfb_module(lfb_module)
        
        if gfb_module is not None:
            self.gfb_module = epic_builder.build_gfb_module(gfb_module)

        if cls_head is not None:
            self.cls_head = builder.build_head(cls_head)
        else:
            print("model without head (may for feature extraction)")

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.append_aux = append_aux

        self.init_weights()

    @property
    def with_lfb_module(self):
        return hasattr(self, 'lfb_module') and self.lfb_module is not None
    
    @property
    def with_gfb_module(self):
        return hasattr(self, 'gfb_module') and self.gfb_module is not None

    @property
    def with_cls_head(self):
        return hasattr(self, 'cls_head') and self.cls_head is not None

    def init_weights(self):
        super(Recognizer, self).init_weights()

        self.backbone.init_weights()
        
        if self.with_lfb_module:
            self.lfb_module.init_weights()
        
        if self.with_gfb_module:
            self.gfb_module.init_weights()

        if self.with_cls_head:
            self.cls_head.init_weights()

    def forward_train(self,
                      num_modalities,
                      img_meta,
                      gt_label,
                      **kwargs): 
        x = kwargs['feature']
        losses = dict()

        if self.append_aux:
            aux = kwargs['auxiliary_feature']
            x = torch.cat([x, aux], 1)

        x = x.reshape((x.shape[0], x.shape[1], 1, 1, 1))
      

        if self.with_lfb_module:
            lfb = kwargs['lfb']
            x = self.lfb_module(x, lfb)
        
        if self.with_gfb_module:
            gfb = kwargs['gfb']
            gfb = dgl.batch(gfb)
            gfb.to(x.device)
            x, gfb_loss = self.gfb_module(x, gfb)

            gfb_loss = self.gfb_module.loss(gfb_loss)
            losses.update(gfb_loss)
        
        x = self.backbone(x)

     
       
        if self.with_cls_head:
            cls_score = self.cls_head(x)
            gt_label = gt_label.squeeze()
            loss_cls = self.cls_head.loss(cls_score, gt_label)
            losses.update(loss_cls)

        return losses

    def forward_test(self,
                     num_modalities,
                     img_meta,
                     **kwargs):
        x = kwargs['feature']

        if self.append_aux:
            aux = kwargs['auxiliary_feature']
            x = torch.cat([x, aux], 1)

        x = x.reshape((x.shape[0], x.shape[1], 1, 1, 1))
        

        if self.with_lfb_module:
            lfb = kwargs['lfb']
            x = self.lfb_module(x, lfb)
        
        if self.with_gfb_module:
            gfb = kwargs['gfb']
            gfb = dgl.batch(gfb)
            gfb.to(x.device)
            x, _ = self.gfb_module(x, gfb)
        
        x = self.backbone(x)

        if self.with_cls_head:
            x = self.cls_head(x)

        ret = [x_i.cpu().numpy() for x_i in x] if isinstance(x, list) else x.cpu().numpy()
        return ret


@RECOGNIZERS.register_module
class Recognizer1(BaseRecognizer):
    """
    perform gfb before lfb (diff in weighting node setting)
    """
    def __init__(self,
                 backbone,
                 append_aux=False,
                 lfb_module=None,
                 gfb_module=None,
                 cls_head=None,
                 train_cfg=None,
                 test_cfg=None):

        super(Recognizer1, self).__init__()

        self.backbone = builder.build_backbone(backbone)

        if lfb_module is not None:
            self.lfb_module = epic_builder.build_lfb_module(lfb_module)
        
        if gfb_module is not None:
            self.gfb_module = epic_builder.build_gfb_module(gfb_module)

        if cls_head is not None:
            self.cls_head = builder.build_head(cls_head)
        else:
            print("model without head (may for feature extraction)")

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.append_aux = append_aux

        self.init_weights()

    @property
    def with_lfb_module(self):
        return hasattr(self, 'lfb_module') and self.lfb_module is not None
    
    @property
    def with_gfb_module(self):
        return hasattr(self, 'gfb_module') and self.gfb_module is not None

    @property
    def with_cls_head(self):
        return hasattr(self, 'cls_head') and self.cls_head is not None

    def init_weights(self):
        super(Recognizer1, self).init_weights()

        self.backbone.init_weights()
        
        if self.with_lfb_module:
            self.lfb_module.init_weights()
        
        if self.with_gfb_module:
            self.gfb_module.init_weights()

        if self.with_cls_head:
            self.cls_head.init_weights()

    def forward_train(self,
                      num_modalities,
                      img_meta,
                      gt_label,
                      **kwargs): 
        x = kwargs['feature']
        losses = dict()

        if self.append_aux:
            aux = kwargs['auxiliary_feature']
            x = torch.cat([x, aux], 1)

        x = x.reshape((x.shape[0], x.shape[1], 1, 1, 1))
      
        
        if self.with_gfb_module:
            gfb = kwargs['gfb']
            gfb = dgl.batch(gfb)
            gfb.to(x.device)
            x, gfb_loss = self.gfb_module(x, gfb)

            gfb_loss = self.gfb_module.loss(gfb_loss)
            losses.update(gfb_loss)
        
        if self.with_lfb_module:
            lfb = kwargs['lfb']
            x = self.lfb_module(x, lfb)
        
        x = self.backbone(x)

     
       
        if self.with_cls_head:
            cls_score = self.cls_head(x)
            gt_label = gt_label.squeeze()
            loss_cls = self.cls_head.loss(cls_score, gt_label)
            losses.update(loss_cls)

        return losses

    def forward_test(self,
                     num_modalities,
                     img_meta,
                     **kwargs):
        x = kwargs['feature']

        if self.append_aux:
            aux = kwargs['auxiliary_feature']
            x = torch.cat([x, aux], 1)

        x = x.reshape((x.shape[0], x.shape[1], 1, 1, 1))
        
        if self.with_gfb_module:
            gfb = kwargs['gfb']
            gfb = dgl.batch(gfb)
            gfb.to(x.device)
            x, _ = self.gfb_module(x, gfb)
        

        if self.with_lfb_module:
            lfb = kwargs['lfb']
            x = self.lfb_module(x, lfb)

        
        x = self.backbone(x)

        if self.with_cls_head:
            x = self.cls_head(x)

        ret = [x_i.cpu().numpy() for x_i in x] if isinstance(x, list) else x.cpu().numpy()
        return ret
