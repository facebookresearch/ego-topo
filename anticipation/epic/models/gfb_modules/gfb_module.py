import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tmodels
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence, PackedSequence

from mmcv.cnn import kaiming_init

from ..registry import GFB_MODULES


@GFB_MODULES.register_module
class SimpleGfbModule(nn.Module):
    def __init__(self, pool_type='avg'):
        super(SimpleGfbModule, self).__init__()

        assert pool_type in ['avg', 'max', 'cur', 'next']
        self.pool_type = pool_type

    def init_weights(self):
        pass

    def forward(self, sfb, g):
        g.to(sfb.device)
        if self.pool_type == 'avg':
            g_feat = dgl.mean_nodes(g, 'feats')
        elif self.pool_type == 'max':
            g_feat = dgl.max_nodes(g, 'feats')
        elif self.pool_type == 'cur':
            g_feat = g.ndata["feats"][g.ndata["cur_status"] == 1]
        elif self.pool_type == 'next':
            #g_feat = g.ndata["feats"][g.ndata["next_status"] == 1]
            g_feat = dgl.mean_nodes(g, 'feats', 'next_status')
        else:
            raise NotImplementedError
        # (B, C) --> (B, C, 1, 1, 1)
        nB, nC = g_feat.size()
        g_feat = g_feat.reshape((nB, nC, 1, 1, 1))

        return torch.cat((sfb, g_feat), dim=1)

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

# GCN + MEAN
@GFB_MODULES.register_module
class GCNGfbModule(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.gcn = GCN(2048, 2048, 2)

    def init_weights(self):
        pass

    def forward(self, sfb, bg):
        bg.to(sfb.device)

        feats, lengths = bg.ndata.pop('feats'), bg.ndata.pop('length')
        feats = feats.sum(1)/lengths.unsqueeze(1).float()
        bg.ndata['feats'] = feats

        feats = self.gcn(bg.ndata.pop('feats'), bg)
        bg.ndata['feats'] = feats
        g_feat = dgl.mean_nodes(bg, 'feats')
      
        # (B, C) --> (B, C, 1, 1, 1)
        nB, nC = g_feat.size()
        g_feat = g_feat.reshape((nB, nC, 1, 1, 1))

        return torch.cat((sfb, g_feat), dim=1)


# # MEAN FEATS
# @GFB_MODULES.register_module
# class GraphX(nn.Module):
#     def __init__(self,**kwargs):
#         super().__init__()

#     def init_weights(self):
#         pass

#     def forward(self, sfb, bg):
#         bg.to(sfb.device)
       
#         feats, lengths = bg.ndata.pop('feats'), bg.ndata.pop('length')

#         feats = feats.sum(1)/lengths.unsqueeze(1).float()
#         bg.ndata['feats'] = feats
#         g_feat = dgl.mean_nodes(bg, 'feats')


#         # (B, C) --> (B, C, 1, 1, 1)
#         nB, nC = g_feat.size()
#         g_feat = g_feat.reshape((nB, nC, 1, 1, 1))

#         return torch.cat((sfb, g_feat), dim=1)


@GFB_MODULES.register_module
class GraphX(nn.Module):
    def __init__(self,**kwargs):
        super().__init__()
        self.rnn = nn.LSTM(2048, 2048, num_layers=2, batch_first=True)
        # self.gcn = GCN(2048, 2048, 1)

        # self.feat = nn.Sequential(
        #         nn.Linear(2048, 2048),
        #         nn.ReLU(True),
        #         nn.Linear(2048, 2048),
        #         nn.ReLU(True))

        # rnet = tmodels.resnet50(pretrained=True)
        # rnet.fc = self.feat
        # self.feat = rnet


    def init_weights(self):
        wts = torch.load('work_dir/pretrain/epoch_50.pth')['state_dict']
        wts = {k.replace('backbone.lstm.',''):v for k,v in wts.items() if 'lstm' in k}
        self.rnn.load_state_dict(wts)
        print ('loaded pretrained LSTM weights')

        # # wts = torch.load('data/affordance/best_model_r50.pth')['model']
        # wts = torch.load('data/affordance/best_model.pth')['model']
        # wts = {k.replace('feat.',''):v for k,v in wts.items() if 'feat' in k}
        # self.feat.load_state_dict(wts)
        # print ('loaded pretrained affordance model weights')


    def named_modules(self, memo=None, prefix=''):
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if name=='rnn' or name=='feat':
                    continue
                if module is None:
                    continue
                submodule_prefix = prefix + ('.' if prefix else '') + name
                for m in module.named_modules(memo, submodule_prefix):
                    yield m

    def forward(self, sfb, bg):
        bg.to(sfb.device)
       
        # # Average all node features first
        # feats, lengths = bg.ndata.pop('feats'), bg.ndata.pop('length')
        # feats = feats.sum(1)/lengths.unsqueeze(1).float()
        # bg.ndata['feats'] = feats

        # # Affordance Model
        # feats = bg.ndata.pop('feats')
        # feats = self.feat(feats)
        # bg.ndata['feats'] = feats

        # # First affordance, then avg output features
        # feats, lengths = bg.ndata.pop('feats'), bg.ndata.pop('length')
        # feats = pack_padded_sequence(feats, lengths, batch_first=True, enforce_sorted=False)
        # feats = PackedSequence(self.feat(feats.data), feats.batch_sizes, feats.sorted_indices, feats.unsorted_indices)
        # feats = pad_packed_sequence(feats, batch_first=True)[0]
        # feats = feats.sum(1)/lengths.unsqueeze(1).float()
        # bg.ndata['feats'] = feats

        # LSTM
        feats, lengths = bg.ndata.pop('feats'), bg.ndata.pop('length')
        packed_feats = pack_padded_sequence(feats, lengths, batch_first=True, enforce_sorted=False)
        packed_output = self.rnn(packed_feats)
        seq, (ht, ct) = packed_output
        bg.ndata['feats'] = ht[-1]

        # # GCN
        # feats = bg.ndata.pop('feats')
        # feats = self.gcn(feats, bg)
        # bg.ndata['feats'] = feats

        # MEAN
        g_feat = dgl.mean_nodes(bg, 'feats')

        # (B, C) --> (B, C, 1, 1, 1)
        nB, nC = g_feat.size()
        g_feat = g_feat.reshape((nB, nC, 1, 1, 1))

        return torch.cat((sfb, g_feat), dim=1)