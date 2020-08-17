import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # g = dgl.batch(g)
        # g.to(sfb.device)
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

        return torch.cat((sfb, g_feat), dim=1), None
    
    def loss(self, loss):
        return {}


@GFB_MODULES.register_module
class WeightedGfbModule(nn.Module):
    def __init__(self, in_channels=2048, node_channels=2048, h_channels=512, next_node_loss=False, loss_weight=1.0, att_act="softmax"):
        super(WeightedGfbModule, self).__init__()

        assert att_act in ["sigmoid", "softmax"]

        self.next_node_loss = next_node_loss
        self.loss_weight = loss_weight

        self.trans_q = nn.Linear(in_channels, h_channels)
        self.trans_v = nn.Linear(node_channels, h_channels)
        self.weight_act = F.sigmoid if att_act == "sigmoid" else F.softmax

    def init_weights(self):
        pass

    def transform_nodes(self, g, q):
        feats = g.ndata['feats']
        v = self.trans_v(feats)

        w = torch.matmul(v, q.view(-1, 1)) / v.shape[1]
        w_norm = self.weight_act(w)

        h = torch.matmul(w_norm.view(1, -1), feats)

        # g.ndata['wfeats'] = h.view(-1)
        # g.ndata["w"] = w 

        return h.view(-1), w
        # return {'wfeats' : h.view(-1), 'w': w}

    def forward(self, sfb, g):
        # g = dgl.batch(g)
        # g.to(sfb.device)

        nB, nC = sfb.size()[:2]
        
        q = self.trans_q(sfb.view(nB, nC))
        # g.ndata["q"] = q
        wfeats = []
        loss = []
        for idx, g_i in enumerate(dgl.unbatch(g)):
            wfeats_i, w = self.transform_nodes(g_i, q[idx])
            wfeats.append(wfeats_i)
            if self.train and self.next_node_loss:
                loss_i = self.node_loss(w, g_i.ndata["next_status"])
                loss.append(loss_i)

        wfeats = torch.stack(wfeats, 0)
        if self.train and self.next_node_loss:
            loss = torch.stack(loss, 0)
        else:
            loss = None
        # g.apply_nodes(self.transform_nodes)

        # wfeats = g.ndata.pop("wfeats")
        # (B, C) --> (B, C, 1, 1, 1)
 
        wfeats = wfeats.reshape((nB, -1, 1, 1, 1))

        return torch.cat((sfb, wfeats), dim=1), loss


    def node_loss(self, w, next_status):
        # pred = nodes.data['w'].view(1, -1)
        # label = torch.argmax(nodes.data["next_status"]).view(1)
        # loss = F.cross_entropy(pred, label)

        # return {"loss": loss}
        pred = w.view(1, -1)
        label = torch.argmax(next_status).view(1)
        loss = F.cross_entropy(pred, label)

        return loss

    def loss(self, loss):
        if not self.next_node_loss:
            return {}
        
        # next_node_loss = []
        # for idx, g_i in enumerate(dgl.unbatch(g)):
        #     pred = g_i.ndata.pop('w').view(1, -1) # (n, 1)
        #     label = troch.argmax(g_i.ndata.pop("next_status")).view(1)
            
        #     gloss = F.cross_entropy(pred, label)
        #     nexxt_node_loss.append(gloss)
        
        # next_node_loss = torch.stack(next_node_loss, 0)
        # g.apply_nodes(self.nodes_loss)
        # next_node_loss = g.ndata["loss"]

        return {"next_node_cls_loss": self.loss_weight * loss}
