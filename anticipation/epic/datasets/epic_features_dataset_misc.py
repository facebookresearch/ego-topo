import bisect
import copy
import os.path as osp
import random
from functools import partial
from collections import defaultdict

import dgl
import numpy as np
import torch
from torch.utils.data import Dataset

import mmcv
from mmcv.parallel import DataContainer as DC
from mmcv.parallel import collate

from .epic_utils import EpicRawFramesRecord, to_tensor, get_visit_feature
import epic.datasets.epic_utils as epic_utils
from .epic_features_dataset import EpicFeatureDatasetGFB


class EpicFeatureDatasetGFBWin(EpicFeatureDatasetGFB):
    '''
    Use precomputed generated graphs in history for each video
    '''
    def __init__(self, gfb_window=40, FPS=60, **kwargs):
        super(EpicFeatureDatasetGFBWin, self).__init__(**kwargs)
        self.gfb_window = 40
        self.FPS = FPS

    def __getitem__(self, idx):
        record = self.video_infos[idx]
        data = super(EpicFeatureDatasetGFB, self).__getitem__(idx)

        self.cur_graph = epic_utils.get_graph(self.graphs[record.path], record.end_frame)
        self.next_graph = epic_utils.get_graph(self.graphs[record.path], record.clip_end_frame) if self.add_next_node else None
        node_feats = self.get_node_feats(self.cur_graph, self.fb[record.path], record.end_frame - self.gfb_window * self.FPS)
        g = self.process_graph_feats(self.cur_graph, node_feats, next_graph=self.next_graph)

        data.update({'gfb':DC(g, stack=False, cpu_only=True)})

        return data
    
    def get_node_feats(self, graph, trunk_features, start_limit):
        node_feats = []
        for node in sorted(list(graph.nodes())):
            visits = graph.node[node]['members']
            if self.node_num_member > 0:
                visits = visits[-self.node_num_member:]
            if self.node_feat_mode == 'last':
                visits = visits[-1:]
            keep = len(visits) - 1
            for i in range(len(visits)):
                if visits[i]['stop'][1] > start_limit:
                    visits[i]['start'] = (visits[i]['start'][0], max(visits[i]['start'][1], start_limit))
                    keep = i
                    break
            visits = visits[keep:]
                # print(visits[i], start_limit)

            feats = [get_visit_feature(trunk_features, v['start'][1], v['stop'][1], self.fb_dim) for v in visits]

            if len(feats) == 0:
                feats = torch.zeros((1, self.fb_dim))
            else:
                feats = torch.cat(feats, 0)
            if self.node_feat_mode == 'avg':
                feats = torch.mean(feats, dim=0, keepdim=True)
            elif self.node_feat_mode == 'max':
                feats, _ = torch.max(feats, dim=0, keepdim=True)

            node_feats.append(feats)

        node_feats = torch.cat(node_feats, 0)  # (N, 2048)

        return node_feats
   

class EpicFeatureDatasetGFBWin1(EpicFeatureDatasetGFBWin):
    '''
    average all features over nodes to mimic LFB
    '''
    def __init__(self, **kwargs):
        super(EpicFeatureDatasetGFBWin1, self).__init__(**kwargs)
    
    def get_visit_feature(self, trunk_features, start, stop, dim):
        feats = []
        for fid in range(start - 31, stop - 31):
            if fid in trunk_features.keys():
                feats.append(to_tensor(trunk_features[fid]))

        # if len(feats) == 0:
        #     return torch.zeros((1, dim))
        # else:
        #     return torch.stack(feats, 0).mean(0, keepdim=True)
        return feats
    
    def get_node_feats(self, graph, trunk_features, start_limit):
        node_feats = []
        for node in sorted(list(graph.nodes())):
            visits = graph.node[node]['members']
            if self.node_num_member > 0:
                visits = visits[-self.node_num_member:]
            if self.node_feat_mode == 'last':
                visits = visits[-1:]
            # may remove all for this node
            keep = len(visits)
            for i in range(len(visits)):
                if visits[i]['stop'][1] > start_limit:
                    visits[i]['start'] = (visits[i]['start'][0], max(visits[i]['start'][1], start_limit))
                    keep = i
                    break
            visits = visits[keep:]
                # print(visits[i], start_limit)

            for v in visits:
                node_feats.extend(self.get_visit_feature(trunk_features, v['start'][1], v['stop'][1], self.fb_dim))

            # if len(feats) == 0:
            #     feats = torch.zeros((1, self.fb_dim))
            # else:
            #     feats = torch.cat(feats, 0)
            # if self.node_feat_mode == 'avg':
            #     feats = torch.mean(feats, dim=0, keepdim=True)
            # elif self.node_feat_mode == 'max':
            #     feats, _ = torch.max(feats, dim=0, keepdim=True)

            # node_feats.extend(feats)
        if len(node_feats) == 0:
            node_feats = torch.zeros((1, self.fb_dim))
        else:
            node_feats = torch.cat(node_feats, 0).mean(dim=0, keepdim=True)

        node_feats = node_feats.expand(len(graph.nodes()), self.fb_dim)

        # node_feats = torch.cat(node_feats, 0)  # (N, 2048)

        return node_feats


class EpicFeatureDatasetGFBWinExtend(EpicFeatureDatasetGFBWin):
    '''
    add window + extend visits to fill the holes
    '''
    def __init__(self, **kwargs):
        super(EpicFeatureDatasetGFBWinExtend, self).__init__(**kwargs)

    def __getitem__(self, idx):
        record = self.video_infos[idx]
        data = super(EpicFeatureDatasetGFB, self).__getitem__(idx)

        self.cur_graph = epic_utils.get_graph(self.graphs[record.path], record.end_frame)
        self.next_graph = epic_utils.get_graph(self.graphs[record.path], record.clip_end_frame) if self.add_next_node else None
        node_feats = self.get_node_feats(self.cur_graph, self.fb[record.path], record.end_frame - self.gfb_window * self.FPS, record.end_frame)
        g = self.process_graph_feats(self.cur_graph, node_feats, next_graph=self.next_graph)

        data.update({'gfb':DC(g, stack=False, cpu_only=True)})

        return data
    
    def get_node_feats(self, graph, trunk_features, start_limit, end_idx):
        node_feats = []
        node_visits = dict()
        for node in sorted(list(graph.nodes())):
            visits = graph.node[node]['members']
            if self.node_num_member > 0:
                visits = visits[-self.node_num_member:]
            if self.node_feat_mode == 'last':
                visits = visits[-1:]
            keep = len(visits) - 1
            for i in range(len(visits)):
                if visits[i]['stop'][1] > start_limit:
                    visits[i]['start'] = (visits[i]['start'][0], max(visits[i]['start'][1], start_limit))
                    keep = i
                    break
            visits = visits[keep:]
            if len(visits) > 1:
                assert (visits[0]['start'][1] >= start_limit)
            
                # print(visits[i], start_limit)
            node_visits[node] = [(v['start'][1], v['stop'][1]) for v in visits]

        idx = [(node, i) for node in sorted(list(graph.nodes())) for i in range(len(node_visits[node]))]
        idx = sorted(idx, key=lambda x:node_visits[x[0]][x[1]][0])
        
        node_visits = self.extend_visits(node_visits, idx, start_limit, end_idx)
        # self.count_missing(node_visits, idx, start_limit, end_idx)

        for node in sorted(list(graph.nodes())):
            feats = [get_visit_feature(trunk_features, v[0], v[1], self.fb_dim) for v in node_visits[node]]

            if len(feats) == 0:
                feats = torch.zeros((1, self.fb_dim))
            else:
                feats = torch.cat(feats, 0)
            if self.node_feat_mode == 'avg':
                feats = torch.mean(feats, dim=0, keepdim=True)
            elif self.node_feat_mode == 'max':
                feats, _ = torch.max(feats, dim=0, keepdim=True)

            node_feats.append(feats)

        node_feats = torch.cat(node_feats, 0)  # (N, 2048)

        return node_feats
    
    def extend_visits(self, node_visits, idx, start_idx, end_idx):
        prev = max(0, start_idx - 1)
        for i in range(len(idx)):
            node_i, visit_i = idx[i]
            v = node_visits[node_i][visit_i]
            if v[0] > prev + 1:
                if i > 0 and node_visits[idx[i - 1][0]][idx[i - 1][1]][1] > start_idx:
                    pre_node, pre_visit = idx[i - 1]
                    m = (prev + v[0]) // 2
                    # print("prev", node_visits[pre_node][pre_visit])
                    node_visits[pre_node][pre_visit] = (node_visits[pre_node][pre_visit][0], m)
                    node_visits[node_i][visit_i] = (m + 1, v[1])
                else:
                    node_visits[node_i][visit_i] = (prev + 1, v[1])
            prev = max(v[1], start_idx - 1)
        
        if prev + 1 < end_idx and node_visits[idx[-1][0]][idx[-1][1]][0] >= start_idx:
            node_i, visit_i = idx[-1]
            node_visits[node_i][visit_i] = (node_visits[node_i][visit_i][0], end_idx)
        
        return node_visits
   
    def count_missing(self, node_visits, idx, start_idx, end_idx):
        count = 0
        miss_visits = []
        prev = max(0, start_idx - 1)
        for i in range(len(idx)):
            v = node_visits[idx[i][0]][idx[i][1]]
            if (v[0] > prev + 1):
                miss_visits.append((prev + 1, v[0] - 1))
            prev = max(v[1], start_idx - 1)
        if prev + 1 < end_idx:
            miss_visits.append((prev + 1, end_idx))
        
        # print(miss_visits)
        for v in miss_visits:
            count += v[1] - v[0] + 1
        
        tot = end_idx - start_idx +1
        print("missing {} frames (rate {}) in {} total frames ({}-{}) with {} missing segments and {} visists".format(count, count / tot, tot, start_idx, end_idx, len(miss_visits), len(idx)))

    
class EpicFeatureDatasetGFBWinExtendRemove(EpicFeatureDatasetGFBWinExtend):
    '''
    add window + extend visits to fill the holes. Remove nodes.
    '''
    def __init__(self, **kwargs):
        super(EpicFeatureDatasetGFBWinExtendRemove, self).__init__(**kwargs)

    def __getitem__(self, idx):
        record = self.video_infos[idx]
        data = super(EpicFeatureDatasetGFB, self).__getitem__(idx)

        self.cur_graph = epic_utils.get_graph(self.graphs[record.path], record.end_frame)
        self.next_graph = epic_utils.get_graph(self.graphs[record.path], record.clip_end_frame) if self.add_next_node else None
        node_feats, node_keep = self.get_node_feats(self.cur_graph, self.fb[record.path], record.end_frame - self.gfb_window * self.FPS, record.end_frame)
        g = self.process_graph_feats(self.cur_graph, node_feats, node_keep, next_graph=self.next_graph)

        data.update({'gfb':DC(g, stack=False, cpu_only=True)})

        return data
    
    def get_node_feats(self, graph, trunk_features, start_limit, end_idx):
        node_feats = []
        node_visits = dict()
        for node in sorted(list(graph.nodes())):
            visits = graph.node[node]['members']
            if self.node_num_member > 0:
                visits = visits[-self.node_num_member:]
            if self.node_feat_mode == 'last':
                visits = visits[-1:]
            keep = len(visits) - 1
            for i in range(len(visits)):
                if visits[i]['stop'][1] > start_limit:
                    visits[i]['start'] = (visits[i]['start'][0], max(visits[i]['start'][1], start_limit))
                    keep = i
                    break
            visits = visits[keep:]
            if len(visits) > 1:
                assert (visits[0]['start'][1] >= start_limit)
            
                # print(visits[i], start_limit)
            node_visits[node] = [(v['start'][1], v['stop'][1]) for v in visits]

        idx = [(node, i) for node in sorted(list(graph.nodes())) for i in range(len(node_visits[node]))]
        idx = sorted(idx, key=lambda x:node_visits[x[0]][x[1]])
        
        node_visits = self.extend_visits(node_visits, idx, start_limit, end_idx)
        # self.count_missing(node_visits, idx, start_limit, end_idx)

        keep = []
        for node in sorted(list(graph.nodes())):
            feats = [get_visit_feature(trunk_features, v[0], v[1], self.fb_dim) for v in node_visits[node]]

            if len(feats) == 0:
                feats = torch.zeros((1, self.fb_dim))
            else:
                feats = torch.cat(feats, 0)
            if self.node_feat_mode == 'avg':
                feats = torch.mean(feats, dim=0, keepdim=True)
            elif self.node_feat_mode == 'max':
                feats, _ = torch.max(feats, dim=0, keepdim=True)

            node_feats.append(feats)

            if len(node_visits[node]) == 1 and node_visits[node][0][1] < start_limit:
                keep.append(False)
            else:
                keep.append(True)

        node_feats = torch.cat(node_feats, 0)  # (N, 2048)

        return node_feats, keep
    
    def process_graph_feats(self, graph, node_feats, node_keep, next_graph=None):
        # -------------------------------------------------------------------#
        # Make the dgl graph now
        cur_node = epic_utils.find_last_visit_node(graph)
        nodes = sorted(list(graph.nodes()))
        node_keep[nodes.index(cur_node)] = True
        nodes = [nodes[i] for i in range(len(nodes)) if node_keep[i]]

        node_feats = node_feats[node_keep]

        node_to_idx = {node: idx for idx, node in enumerate(nodes) }
        # src, dst = [], []
        # if len(graph.edges()) > 0:
        #     src, dst = zip(*graph.edges())
        #     src = [node_to_idx[node] for node in src]
        #     dst = [node_to_idx[node] for node in dst]

        g = dgl.DGLGraph()
        g.add_nodes(len(nodes))
        # g.add_edges(src, dst)
        # g.add_edges(dst, src)  # undirected
        g.add_edges(g.nodes(), g.nodes())  # add self loops

        g.ndata['feats'] = node_feats

        cur_status = torch.zeros(len(nodes))
        # cur_node = 0 if graph.last_state['node'] is None else node_to_idx[graph.last_state['node']]
        cur_status[node_to_idx[cur_node]] = 1
        g.ndata['cur_status'] = cur_status

        if next_graph is not None:
            next_status = torch.zeros(len(nodes))
            next_node = epic_utils.find_last_visit_node(next_graph)
            if next_node not in node_to_idx:
                next_node = cur_node
            next_status[node_to_idx[next_node]] = 1
            g.ndata["next_status"] = next_status

            if next_node == cur_node:
                next1_status = next_status
            else:
                next1_status = 1 - cur_status
            g.ndata["next1_status"] = next1_status

        return g


class EpicFeatureDatasetGFBWinExtendNew(EpicFeatureDatasetGFBWinExtend):
    '''
    add window + extend visits to fill the holes and update get_visit_feature
    '''
    def __init__(self, **kwargs):
        super(EpicFeatureDatasetGFBWinExtendNew, self).__init__(**kwargs)

    def get_node_feats(self, graph, trunk_features, start_limit, end_idx):
        node_feats = []
        node_visits = dict()
        for node in sorted(list(graph.nodes())):
            visits = graph.node[node]['members']
            if self.node_num_member > 0:
                visits = visits[-self.node_num_member:]
            if self.node_feat_mode == 'last':
                visits = visits[-1:]
            keep = len(visits) - 1
            for i in range(len(visits)):
                if visits[i]['stop'][1] > start_limit:
                    visits[i]['start'] = (visits[i]['start'][0], max(visits[i]['start'][1], start_limit))
                    keep = i
                    break
            visits = visits[keep:]
            if len(visits) > 1:
                assert (visits[0]['start'][1] >= start_limit)
            
                # print(visits[i], start_limit)
            node_visits[node] = [(v['start'][1], v['stop'][1]) for v in visits]

        idx = [(node, i) for node in sorted(list(graph.nodes())) for i in range(len(node_visits[node]))]
        idx = sorted(idx, key=lambda x:node_visits[x[0]][x[1]])
        
        node_visits = self.extend_visits(node_visits, idx, start_limit, end_idx)
        # self.count_missing(node_visits, idx, start_limit, end_idx)

        for node in sorted(list(graph.nodes())):
            feats = []
            for v in node_visits[node]:
                v_feat = self.get_visit_feature(trunk_features, v[0], v[1], self.fb_dim, end_idx)
                if v_feat is not None:
                    feats.append(v_feat)

            if len(feats) == 0:
                feats = torch.zeros((1, self.fb_dim))
            else:
                feats = torch.cat(feats, 0)
            if self.node_feat_mode == 'avg':
                feats = torch.mean(feats, dim=0, keepdim=True)
            elif self.node_feat_mode == 'max':
                feats, _ = torch.max(feats, dim=0, keepdim=True)

            node_feats.append(feats)

        node_feats = torch.cat(node_feats, 0)  # (N, 2048)

        return node_feats
    
    def get_visit_feature(self, trunk_features, start, stop, dim, end_idx):
        feats = []
        for fid in range(start, min(stop, end_idx - 31)):
            if fid in trunk_features.keys():
                feats.append(to_tensor(trunk_features[fid]))

        if len(feats) == 0:
            return None
        else:
            return torch.stack(feats, 0).mean(0, keepdim=True)
    

class EpicFeatureDatasetGFBWinExtendRemoveNew(EpicFeatureDatasetGFBWinExtendRemove):
    '''
    add window + extend visits to fill the holes, remove old nodes,  update get_visit_feature. 
    '''
    def __init__(self, **kwargs):
        super(EpicFeatureDatasetGFBWinExtendRemoveNew, self).__init__(**kwargs)

    def get_node_feats(self, graph, trunk_features, start_limit, end_idx):
        node_feats = []
        node_visits = dict()
        for node in sorted(list(graph.nodes())):
            visits = graph.node[node]['members']
            if self.node_num_member > 0:
                visits = visits[-self.node_num_member:]
            if self.node_feat_mode == 'last':
                visits = visits[-1:]
            keep = len(visits) - 1
            for i in range(len(visits)):
                if visits[i]['stop'][1] > start_limit:
                    visits[i]['start'] = (visits[i]['start'][0], max(visits[i]['start'][1], start_limit))
                    keep = i
                    break
            visits = visits[keep:]
            if len(visits) > 1:
                assert (visits[0]['start'][1] >= start_limit)
            
                # print(visits[i], start_limit)
            node_visits[node] = [(v['start'][1], v['stop'][1]) for v in visits]

        idx = [(node, i) for node in sorted(list(graph.nodes())) for i in range(len(node_visits[node]))]
        idx = sorted(idx, key=lambda x:node_visits[x[0]][x[1]])
        
        node_visits = self.extend_visits(node_visits, idx, start_limit, end_idx)
        # self.count_missing(node_visits, idx, start_limit, end_idx)

        keep = []
        for node in sorted(list(graph.nodes())):
            feats = []
            for v in node_visits[node]:
                v_feat = self.get_visit_feature(trunk_features, v[0], v[1], self.fb_dim, end_idx)
                if v_feat is not None:
                    feats.append(v_feat)

            if len(feats) == 0:
                feats = torch.zeros((1, self.fb_dim))
            else:
                feats = torch.cat(feats, 0)
            if self.node_feat_mode == 'avg':
                feats = torch.mean(feats, dim=0, keepdim=True)
            elif self.node_feat_mode == 'max':
                feats, _ = torch.max(feats, dim=0, keepdim=True)

            node_feats.append(feats)

            if len(node_visits[node]) == 1 and node_visits[node][0][1] < start_limit:
                keep.append(False)
            else:
                keep.append(True)

        node_feats = torch.cat(node_feats, 0)  # (N, 2048)

        return node_feats, keep
    
    def get_visit_feature(self, trunk_features, start, stop, dim, end_idx):
        feats = []
        for fid in range(start, min(stop, end_idx - 31)):
            if fid in trunk_features.keys():
                feats.append(to_tensor(trunk_features[fid]))

        if len(feats) == 0:
            return None
        else:
            return torch.stack(feats, 0).mean(0, keepdim=True)


class EpicFeatureDatasetGFBWinExtendRemoveNew1(EpicFeatureDatasetGFBWinExtendRemoveNew):
    '''
    add window + extend visits to fill the holes, remove old nodes,  update get_visit_feature. 
    '''
    def __init__(self, **kwargs):
        super(EpicFeatureDatasetGFBWinExtendRemoveNew1, self).__init__(**kwargs)

    def get_node_feats(self, graph, trunk_features, start_limit, end_idx):
        # print("-----------------------")
        # print(start_limit, end_idx)
        node_feats = []
        node_visits = dict()
        for node in sorted(list(graph.nodes())):
            visits = graph.node[node]['members']
            if self.node_num_member > 0:
                visits = visits[-self.node_num_member:]
            if self.node_feat_mode == 'last':
                visits = visits[-1:]
            keep = len(visits) - 1
            for i in range(len(visits)):
                if visits[i]['stop'][1] >= start_limit:
                    visits[i]['start'] = (visits[i]['start'][0], max(visits[i]['start'][1], start_limit))
                    keep = i
                    break
            # print("ori", visits)
            visits = visits[keep:]
            # print("keep", visits)
            if len(visits) > 1:
                assert (visits[0]['start'][1] >= start_limit)
            
                # print(visits[i], start_limit)
            node_visits[node] = [(v['start'][1], v['stop'][1]) for v in visits]

        idx = [(node, i) for node in sorted(list(graph.nodes())) for i in range(len(node_visits[node]))]
        idx = sorted(idx, key=lambda x:node_visits[x[0]][x[1]])
        
        node_visits = self.extend_visits(node_visits, idx, start_limit, end_idx)
        # self.count_missing(node_visits, idx, start_limit, end_idx)

        keep = []
        lfb_ids = []
        for node in sorted(list(graph.nodes())):
            feats = []
            ids = []
            for v in node_visits[node]:
                v_feat, id_feat = self.get_visit_feature(trunk_features, v[0], v[1], self.fb_dim, end_idx)
                if v_feat is not None:
                    feats.extend(v_feat)
                    ids.extend(id_feat)

            # if len(feats) == 0:
            #     feats = torch.zeros((1, self.fb_dim))
            # else:
            #     feats = torch.cat(feats, 0)
            # if self.node_feat_mode == 'avg':
            #     feats = torch.mean(feats, dim=0, keepdim=True)
            # elif self.node_feat_mode == 'max':
            #     feats, _ = torch.max(feats, dim=0, keepdim=True)

            

            if len(node_visits[node]) == 1 and node_visits[node][0][1] < start_limit:
                keep.append(False)
            else:
                keep.append(True)
                node_feats.extend(feats)
                lfb_ids.extend(ids)
                # print("node", node_visits[node])
                assert(node_visits[node][0][0] >= start_limit)
                # print("ids", ids)


        if len(node_feats) == 0:
            node_feats = torch.zeros((1, self.fb_dim))
        else:
            node_feats = torch.cat(node_feats, 0).mean(dim=0, keepdim=True)
        
        # print(len(lfb_ids), sorted(lfb_ids), start_limit, end_idx)

        node_feats = node_feats.expand(len(graph.nodes()), self.fb_dim)

        return node_feats, keep
    
    def get_visit_feature(self, trunk_features, start, stop, dim, end_idx):
        feats = []
        ids = []
        for fid in range(start, min(stop, end_idx - 31)):
            if fid in trunk_features.keys():
                feats.append(to_tensor(trunk_features[fid]))
                ids.append(fid)

        # if len(feats) == 0:
        #     return None
        # else:
        #     return torch.stack(feats, 0).mean(0, keepdim=True)

        return feats, ids