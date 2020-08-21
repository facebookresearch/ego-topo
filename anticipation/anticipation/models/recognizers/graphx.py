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

import dgl
import dgl.function as fn
from dgl.nn.pytorch import GraphConv
class GCN(nn.Module):
    def __init__(self, in_dim, h_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([GraphConv(in_dim, h_dim, F.relu)])
        for l in range(1, num_layers):
            self.layers.append(GraphConv(h_dim, h_dim, F.relu))

    def forward(self, inputs, g):
        h = inputs
        for layer in self.layers:
            h = layer(h, g)
        return h



from anticipation.models import nl

@RECOGNIZERS.register_module
class GraphX(BaseRecognizer):
    def __init__(self, backbone, cls_head, train_cfg=None, test_cfg=None):
        super().__init__()
        self.backbone = builder.build_backbone(backbone)
        self.cls_head = builder.build_head(cls_head)
        self.aux_head = builder.build_head(cls_head)

        # self.vdist = nn.Linear(2048, 125)
        # self.ndist = nn.Linear(2048, 352)

        # self.rnn = nn.LSTM(2048, 2048, num_layers=2, batch_first=True)
        self.gcn = GCN(2048, 2048, 1)

        # self.mlp = nn.Sequential(
        #             nn.Linear(2048, 2048),
        #             nn.ReLU(True),
        #             nn.Linear(2048, 2048),
        #             nn.ReLU(True),
        #             nn.Linear(2048, 2048),
        #             nn.ReLU(True),
        #             )

        # self.current_node_embed = nn.Embedding(3, 256)
        # self.squash = nn.Sequential(
        #                 nn.Linear(2048+256, 2048),
        #                 nn.ReLU(True),
                    #     nn.Linear(2048, 2048),
                    #     nn.ReLU(True),
                    #     nn.Linear(2048, 2048),
                    #     nn.ReLU(True),
                    # )

        self.lfb_nl = nl.LFB_NL(2048, 2048, 2048, 1)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights()


    def init_weights(self):
        super().init_weights()
        self.backbone.init_weights()
        self.cls_head.init_weights()
        self.aux_head.init_weights()

        # wts = torch.load('work_dir/pretrain/latest.pth')['state_dict']
        # wts = {k.replace('rnn.',''):v for k,v in wts.items() if 'rnn' in k}
        # self.rnn.load_state_dict(wts)
        # print ('loaded pretrained LSTM weights')

        # for param in self.rnn.parameters():
        #     param.requires_grad = False


        # wts = torch.load('work_dir/alt_graph_ce/epoch_100_ce_mlp.pth')['state_dict']
        # # wts = torch.load('work_dir/pretrain/epoch_20.pth')['state_dict']
        # wts = {k.replace('mlp.',''):v for k,v in wts.items() if 'mlp' in k}
        # self.mlp.load_state_dict(wts)
        # print ('loaded pretrained MLP weights')

        # for param in self.mlp.parameters():
        #     param.requires_grad = False


    def parameters(self):
        output = filter(lambda p: p.requires_grad, super().parameters())
        return output


    def get_graph_feat(self, sfb, bg):
        bg.to(sfb.device)


        losses = {}

        # # LSTM
        # feats, lengths = bg.ndata.pop('feats'), bg.ndata.pop('length')
        # packed_feats = pack_padded_sequence(feats, lengths, batch_first=True, enforce_sorted=False)
        # packed_output = self.rnn(packed_feats)
        # seq, (ht, ct) = packed_output
        # bg.ndata['feats'] = ht[-1]


        # Average all node features + MLP
        feats, lengths = bg.ndata.pop('feats'), bg.ndata.pop('length')
        feats = feats.sum(1)/lengths.unsqueeze(1).float()
        # feats = self.mlp(feats)
        bg.ndata['feats'] = feats


        # # REMOVE THE CURRENT NODE FEATURES
        # current_mask = (bg.ndata['cur_status']==1).float()
        # bg.ndata['feats'] = bg.ndata['feats'] * (1-current_mask.unsqueeze(1))


        # # DIST EMBED then fc back to right dim
        # feats = bg.ndata.pop('feats')
        # feats = torch.cat([feats, self.current_node_embed(bg.ndata.pop('cur_status').long())], 1)
        # feats = self.squash(feats)
        # bg.ndata['feats'] = feats


        # # GCN
        # feats = bg.ndata.pop('feats')
        # feats = self.gcn(feats, bg)
        # bg.ndata['feats'] = feats

        # MEAN
        g_feat = dgl.mean_nodes(bg, 'feats')

        # # NONLOCAL
        # gfb = [g.ndata.pop('feats') for g in dgl.unbatch(bg)]
        # gfb = pad_sequence(gfb, batch_first=True) # (B, Nmax, 2048)

        # # g_feat = self.lfb_nl(sfb.squeeze(), gfb)

        # g_feat, nl_loss = self.lfb_nl(sfb.squeeze(), gfb, bg)
        # # losses.update({'nl_loss':nl_loss})


        # Post processing
        nB, nC = g_feat.size()
        g_feat = g_feat.reshape((nB, nC, 1, 1, 1))

        

        return g_feat, losses


    def forward_train(self, num_modalities, img_meta, gt_label, **kwargs): 
        losses = dict()

        x = kwargs['feature']
        x = x.reshape((x.shape[0], x.shape[1], 1, 1, 1))
        
        #---------------------------------------------------------------------------------------#

        g_feat, g_loss = self.get_graph_feat(x, kwargs['gfb'])
        losses.update(g_loss)

        # aux_pred = self.aux_head(g_feat)
        # loss_aux = self.aux_head.loss(aux_pred, gt_label.squeeze())
        # loss_aux = {re.sub('task_[0-9]', 'aux', k):v for k, v in loss_aux.items()}
        # losses.update(loss_aux)

        #---------------------------------------------------------------------------------------#

        x = torch.cat([x, g_feat], 1)
        x = self.backbone(x)
        cls_score = self.cls_head(x)
        gt_label = gt_label.squeeze()
        loss_cls = self.cls_head.loss(cls_score, gt_label)
        losses.update(loss_cls)

        return losses

    def forward_test(self, num_modalities, img_meta, **kwargs):
        x = kwargs['feature']
        x = x.reshape((x.shape[0], x.shape[1], 1, 1, 1))
        
        g_feat, _ = self.get_graph_feat(x, kwargs['gfb'])
        
        x = torch.cat([x, g_feat], 1)
        x = self.backbone(x)
        x = self.cls_head(x)

        ret = [x_i.cpu().numpy() for x_i in x] if isinstance(x, list) else x.cpu().numpy()
        return ret