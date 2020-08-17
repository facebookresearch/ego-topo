import bisect
import copy
import os.path as osp
import random
from functools import partial
import os 

import dgl
import numpy as np
import torch
from torch.utils.data import Dataset
import collections
import itertools
import copy
import tqdm
from torchvision import transforms
from PIL import Image
import networkx as nx

import mmcv
from mmcv.parallel import DataContainer as DC
from functools import lru_cache

from .epic_utils import EpicRawFramesRecord, to_tensor
import epic.datasets.epic_utils as epic_utils
from .epic_future_labels import EpicFutureLabelsGFB1, EpicFutureLabelsGFB


class EpicFutureLabelsGFB2(EpicFutureLabelsGFB):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_visit_feature(self, trunk_features, start, stop, dim):
        inds = []
        for fid in range(start - 31, stop - 31):
            if fid in trunk_features.keys():
                inds.append(fid)

        if len(inds)==0:
            return None

        # if self.test_mode:
        #     # f_id = inds[len(inds)//2]
        #     feats = [to_tensor(trunk_features[fid]) for fid in inds]
        
        #     feat = torch.stack(feats, 0).mean(0, keepdim=False)
        # else:
        #     f_id = inds[np.random.randint(len(inds))]

        #     feat = trunk_features[f_id]
        #     feat = to_tensor(feat)
        
        feats = [to_tensor(trunk_features[fid]) for fid in inds]
        
        feat = torch.stack(feats, 0).mean(0, keepdim=False)
        
        return feat


class EpicFutureLabelsGFBNode(EpicFutureLabelsGFB1):

    def __init__(self, node_num_member=8, rand_visit=False, **kwargs):
        super().__init__(**kwargs)

        self.node_num_member = node_num_member
        self.rand_visit = rand_visit

    def get_node_feats(self, graph, trunk_features):
        node_feats = []
        node_length = []

        for node in sorted(graph.nodes()):
            visits = graph.node[node]['members']

            feats = []
            for visit in visits:
                v_feat = self.get_visit_feature(trunk_features, visit['start'][1], visit['stop'][1], self.fb_dim)
                if v_feat is not None:
                    feats.append(v_feat)
            feats = feats or [torch.zeros(self.fb_dim)]

            # # Pick last self.node_num_member visits
            # feats = feats[-self.node_num_member:]
            # length = len(feats)
            # feats = feats + [torch.zeros(self.fb_dim)]*(self.node_num_member-length)

            # uniformly pick self.node_num_member visits
            if self.node_num_member > 0:
                if self.rand_visit and len(feats) > self.node_num_member:
                    idxes = sorted(np.random.choice(len(feats), self.node_num_member, replace=False))
                else:
                    idxes = np.round(np.linspace(0, len(feats) - 1, self.node_num_member)).astype(int)
                feats = [feats[idx] for idx in idxes]
            else:
                feats = [torch.stack(feats, 0).mean(0)]
            length = len(feats)

            feats = torch.stack(feats, 0) # (K, 2048)
            node_feats.append(feats)
            node_length.append(length)
            
        node_feats = torch.stack(node_feats, 0)  # (N, K, 2048)
        node_length = torch.LongTensor(node_length)

        return node_feats, node_length
    

class EpicFutureLabelsGFBExtend(EpicFutureLabelsGFB1):
    """
    extend visits to fill holes
    """
    def __init__(self, node_num_member=8, **kwargs):
        super().__init__(**kwargs)

        self.node_num_member = node_num_member

    def get_node_feats(self, graph, trunk_features):
        node_feats = []
        node_length = []
        node_visits = dict()
        for node in sorted(graph.nodes()):
            visits = graph.node[node]['members']

            node_visits[node] = [(v['start'][1], v['stop'][1]) for v in visits]
        
        idx = [(node, i) for node in sorted(list(graph.nodes())) for i in range(len(node_visits[node]))]
        idx = sorted(idx, key=lambda x:node_visits[x[0]][x[1]][0])
        # print("node", node_visits)
        # print("idx", idx)
        node_visits = self.extend_visits(node_visits, idx)
        # print("after", node_visits)
        for node in sorted(graph.nodes()):
            visits = node_visits[node]
            feats = []
            for visit in visits:
                v_feat = self.get_visit_feature(trunk_features, visit[0], visit[1], self.fb_dim)
                if v_feat is not None:
                    feats.append(v_feat)
            feats = feats or [torch.zeros(self.fb_dim)]

            # # Pick last self.node_num_member visits
            # feats = feats[-self.node_num_member:]
            # length = len(feats)
            # feats = feats + [torch.zeros(self.fb_dim)]*(self.node_num_member-length)

            # uniformly pick self.node_num_member visits
            if self.node_num_member > 0:
                feats = [feats[idx] for idx in np.round(np.linspace(0, len(feats) - 1, self.node_num_member)).astype(int)]
            else:
                feats = [torch.stack(feats, 0).mean(0)]
            length = len(feats)

            feats = torch.stack(feats, 0) # (K, 2048)
            node_feats.append(feats)
            node_length.append(length)
            
        node_feats = torch.stack(node_feats, 0)  # (N, K, 2048)
        node_length = torch.LongTensor(node_length)

        return node_feats, node_length
    
    def extend_visits(self, node_visits, idx):
        prev = node_visits[idx[0][0]][idx[0][1]][1]
        for i in range(1, len(idx)):
            node_i, visit_i = idx[i]
            v = node_visits[node_i][visit_i]
            if v[0] > prev + 1:
                pre_node, pre_visit = idx[i - 1]
                m = (prev + v[0]) // 2
                node_visits[pre_node][pre_visit] = (node_visits[pre_node][pre_visit][0], m)
                node_visits[node_i][visit_i] = (m + 1, v[1])
            prev = v[1]
        
        return node_visits


class EpicFutureLabelsGFB3(EpicFutureLabelsGFB):

    def __init__(self, visit_num=3, **kwargs):
        super().__init__(**kwargs)

        self.visit_num = visit_num

    def get_visit_feature(self, trunk_features, start, stop, dim):
        inds = []
        for fid in range(start - 31, stop - 31):
            if fid in trunk_features.keys():
                inds.append(fid)

        if len(inds)==0:
            return None

        # if self.test_mode:
        #     # f_id = inds[len(inds)//2]
        #     feats = [to_tensor(trunk_features[fid]) for fid in inds]
        #     feats = [
        #         to_tensor(trunk_features[inds[idx]])
        #         for idx in np.round(np.linspace(0, len(inds) - 1, self.visit_num)).astype(int)
        #     ]
        
        #     # feat = torch.stack(feats, 0).mean(0, keepdim=False)
        # else:
        #     f_id = inds[np.random.randint(len(inds))]

        #     feat = trunk_features[f_id]
        #     feat = to_tensor(feat)
        
        # feats = [to_tensor(trunk_features[fid]) for fid in inds]
        feats = [to_tensor(trunk_features[fid]) for fid in inds]
        feats = [
            to_tensor(trunk_features[inds[idx]])
            for idx in np.round(np.linspace(0, len(inds) - 1, self.visit_num)).astype(int)
        ]
        
        feat = torch.stack(feats, 0).mean(0, keepdim=False)
        
        return feat


class EpicFutureLabelsGFB4(EpicFutureLabelsGFB):

    def __init__(self, visit_num=3, **kwargs):
        super().__init__(**kwargs)

        self.visit_num = visit_num

    def get_visit_feature(self, trunk_features, start, stop, dim):
        inds = []
        for fid in range(start - 31, stop - 31):
            if fid in trunk_features.keys():
                inds.append(fid)

        if len(inds)==0:
            return None

        if self.test_mode:
            # f_id = inds[len(inds)//2]
            feats = [to_tensor(trunk_features[fid]) for fid in inds]
        
            # feat = torch.stack(feats, 0).mean(0, keepdim=False)
        else:
            feats = [
                to_tensor(trunk_features[inds[idx]])
                for idx in np.round(np.linspace(0, len(inds) - 1, self.visit_num)).astype(int)
            ]
            # feat = to_tensor(feat)
        
        # feats = [to_tensor(trunk_features[fid]) for fid in inds]
        # feats = [to_tensor(trunk_features[fid]) for fid in inds]
        
        
        feat = torch.stack(feats, 0).mean(0, keepdim=False)
        
        return feat


class EpicFutureLabelsGFB5(EpicFutureLabelsGFB):

    def __init__(self, visit_num=3, **kwargs):
        super().__init__(**kwargs)

        self.visit_num = visit_num

    def get_visit_feature(self, trunk_features, start, stop, dim):
        inds = []
        for fid in range(start - 31, stop - 31):
            if fid in trunk_features.keys():
                inds.append(fid)

        if len(inds)==0:
            return None

        if self.test_mode:
            # f_id = inds[len(inds)//2]
            feats = [to_tensor(trunk_features[fid]) for fid in inds]
        
            # feat = torch.stack(feats, 0).mean(0, keepdim=False)
        else:
            feats = [
                to_tensor(trunk_features[inds[idx]])
                for idx in np.random.choice(len(inds), self.visit_num)
            ]
            # feat = to_tensor(feat)
        
        # feats = [to_tensor(trunk_features[fid]) for fid in inds]
        # feats = [to_tensor(trunk_features[fid]) for fid in inds]
        
        
        feat = torch.stack(feats, 0).mean(0, keepdim=False)
        
        return feat


class EpicFutureLabelsGFBVisit(EpicFutureLabelsGFB):

    def __init__(self, node_num_member=8, rand_visit=False, **kwargs):
        super().__init__(**kwargs)

        self.node_num_member = node_num_member
        self.rand_visit = rand_visit
    
    def get_visit_feature(self, trunk_features, start, stop, dim):
        inds = []
        for fid in range(start - 31, stop - 31):
            if fid in trunk_features.keys():
                inds.append(fid)

        return inds
        # if len(inds)==0:
        #     return None

        # if self.test_mode:
        #     f_id = inds[len(inds)//2]
        #     feats = [to_tensor(trunk_features[fid]) for fid in inds]
        
        #     feat = torch.stack(feats, 0).mean(0, keepdim=False)
        # else:
        #     f_id = inds[np.random.randint(len(inds))]

        #     feat = trunk_features[f_id]
        #     feat = to_tensor(feat)
        
        # return feat

    def get_node_feats(self, graph, trunk_features):
        node_feats = []
        node_length = []

        for node in sorted(graph.nodes()):
            visits = graph.node[node]['members']

            feats_idx = []
            for visit in visits:
                v_feat_idx = self.get_visit_feature(trunk_features, visit['start'][1], visit['stop'][1], self.fb_dim)
                feats_idx.extend(v_feat_idx)
            #         feats.append(v_feat)
            # feats = feats or [torch.zeros(self.fb_dim)]

            # # Pick last self.node_num_member visits
            # feats = feats[-self.node_num_member:]
            # length = len(feats)
            # feats = feats + [torch.zeros(self.fb_dim)]*(self.node_num_member-length)

            # uniformly pick self.node_num_member visits
            if self.node_num_member > 0:
                if self.rand_visit and len(feats_idx) > self.node_num_member:
                    idxes = sorted(np.random.choice(feats_idx, self.node_num_member, replace=False))
                    feats = [to_tensor(trunk_features[idx]) for idx in idxes]
                else:
                    if len(feats_idx) > 0:
                        idxes_ = np.round(np.linspace(0, len(feats_idx) - 1, self.node_num_member)).astype(int)
                        # print(idxes_, len(feats_idx))
                        idxes = [feats_idx[i] for i in idxes_]
                        feats = [to_tensor(trunk_features[idx]) for idx in idxes]
                    else:
                        feats = [torch.zeros(self.fb_dim)] * self.node_num_member
                
            else:
                if len(feast_idx) > 0:
                    feats = [to_tensor(trunk_features[idx]) for idx in feats_idx]
                    feats = [torch.stack(feats, 0).mean(0)]
                else:
                    feats = [torch.zeros(self.fb_dim)]
            
            length = len(feats)

            feats = torch.stack(feats, 0) # (K, 2048)
            node_feats.append(feats)
            node_length.append(length)
            
        node_feats = torch.stack(node_feats, 0)  # (N, K, 2048)
        node_length = torch.LongTensor(node_length)

        return node_feats, node_length


class EpicFutureLabelsGFBAug(EpicFutureLabelsGFBVisit):
    """
    add graph_augmentation
    """
    def __init__(self, graph_aug=False, graph_drop=0.5, **kwargs):
        super().__init__(**kwargs)

        self.graph_aug = graph_aug
        self.graph_drop = graph_drop
    
    def graph_augmentation(self, G, keep):

        # IMPORTANT + FOR MEMORY USAGE If you use an RNN backbone!!!!
            # node dropout
        p = self.graph_drop
        nodes = keep.nonzero().view(-1).tolist()
        if len(nodes)>1:
            node_drop = torch.rand(len(nodes))
            node_drop = (node_drop>p).float()
            while node_drop.sum()==len(nodes):
                node_drop = torch.rand(len(nodes))
                node_drop = (node_drop>p)
            for i in node_drop.nonzero().view(-1).tolist():
                keep[nodes[i]] = 0
        
        return keep
       
    def process_graph_feats(self, graph, trunk_features, future_labels):
        graph = copy.deepcopy(graph)
        # Drop useless visits (VERY IMPORTANT FOR GTEA's BLACK FRAMES !!!!!)
        keep = torch.ones((len(graph.nodes())))
        node_feats, node_length = self.get_node_feats(graph, trunk_features)
        if self.dset=='gtea':
            for i, node in enumerate(sorted(graph.nodes())):
                visits = graph.node[node]['members']
                if len(visits)==1 and visits[0]['stop'][1]-visits[0]['start'][1]<self.fps:
                    # graph.remove_node(node)
                    keep[i] = 0

        if not self.test_mode and self.graph_aug:
            keep = self.graph_augmentation(graph, keep)
        
        nodes = sorted(graph.nodes())
        for i in range(keep.shape[0]):
            if keep[i] == 0:
                graph.remove_node(nodes[i])
        # -------------------------------------------------------------------#
        # Make the dgl graph now
        nodes = sorted(graph.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        src, dst = [], []
        if len(graph.edges()) > 0:
            src, dst = zip(*graph.edges())
            src = [node_to_idx[node] for node in src]
            dst = [node_to_idx[node] for node in dst]

        g = dgl.DGLGraph()
        g.add_nodes(len(nodes))
        g.add_edges(src, dst)
        g.add_edges(dst, src)  # undirected
        g.add_edges(g.nodes(), g.nodes())  # add self loops

        g.ndata['feats'] = node_feats[keep == 1]
        g.ndata['length'] = node_length[keep == 1]

        if self.label=='int':
            g.ndata['labels'] = future_labels['ints'][keep == 1]
        elif self.label=='noun':
            g.ndata['labels'] = future_labels['nouns'][keep == 1]
        elif self.label == 'verb':
            g.ndata['labels'] = future_labels['verbs'][keep == 1]

        cur_status = torch.zeros(len(nodes))
        cur_node = epic_utils.find_last_visit_node(graph)
        cur_status[node_to_idx[cur_node]] = 1
        g.ndata['cur_status'] = cur_status

        nbhs = nx.ego_graph(graph, cur_node, radius=2, center=False).nodes()
        for nbh in nbhs:
            cur_status[node_to_idx[nbh]] = 2
        g.ndata['cur_status'] = cur_status

        return g