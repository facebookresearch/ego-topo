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
from .epic_features_dataset import EpicFeatureDataset


def pad_seq_feats(feats, labels, max_length, dim=2048, append_zero=False):
    length = len(feats) - 1 if not append_zero else len(feats)
    feats = torch.cat(feats, 0)
    if labels is not None:
        labels = torch.LongTensor(labels)
    
    if feats.shape[0] <= max_length:
        pad_feats = torch.zeros((max_length + 1, dim))
        pad_feats[:feats.shape[0]] = feats
        feats = pad_feats
        if labels is not None:
            pad_labels = -torch.ones((max_length + 1, labels.shape[1]), dtype=torch.long)
            pad_labels[:labels.shape[0]]= labels
            labels = pad_labels
    
    if length == 0:
        length = 1
    
    return feats, labels, length

class EpicNodeDataset(Dataset):
    def __init__(self, graph_root, fb, ann_file=None, add_verb=False, add_noun=False, fb_dim=2048, max_length=10):

        self.fb = mmcv.load(fb)
        self.fb_dim = fb_dim
        self.max_length = max_length
        self.add_verb = add_verb
        self.add_noun = add_noun
        self.ann_file = ann_file

        videos = list(self.fb.keys())
        self.graphs = self.load_graphs(graph_root, videos)

        self.data = self.build_dataset()
        # just for mmaction sampler interface
        self.flag = np.ones(len(self), dtype=np.uint8)

    def load_graphs(self, graph_root, videos):
        graphs = {}
        for vid in videos:
            graphs[vid] = torch.load("{}/{}_graph.pth".format(graph_root, vid[4:]))

        print("Finished pre-loading graphs")
        return graphs      

    def __len__(self):
        return len(self.data)
    
    def get_annos(self, records):
        annos = defaultdict(list)
        start_frames = defaultdict(list)
        for r in records:
            annos[r.path].append(r)
        
        for k in annos.keys():
            annos[k] = sorted(annos[k], key=lambda x: x.start_frame)
            start_frames[k] = [r.start_frame for r in annos[k]]
        
        return annos, start_frames

    def build_dataset(self):
        annos = None
        if self.add_verb or self.add_noun:
            annos, start_frames = self.get_annos([EpicRawFramesRecord(x.strip().split('\t')) for x in open(self.ann_file)])
            print("Get {} annos.".format(len(annos.keys())))

        records = []
        for vid, graph_history in self.graphs.items():
            frames, graphs = graph_history["frames"], graph_history["graphs"]

            G = graphs[-1]['G']
            for node in sorted(list(G.nodes())):
                visits = G.node[node]['members']
                visits = [{'start': frames[visit['start']], 'stop': frames[visit['stop']]} for visit in visits]
                if annos is not None:
                    labels = [epic_utils.get_visit_labels(annos[vid], start_frames[vid], v['start'][1], v['stop'][1]) for v in visits]
                for i in range(min(self.max_length, len(visits) - 1), len(visits)):
                    s, e = max(0, i - self.max_length), i + 1
                    records.append((vid, visits[s:e], labels[s:e] if annos is not None else None))
        print("finish building {} samples".format(len(records)))

        return records

    def __getitem__(self, idx):
        vid, visits, _labels = self.data[idx]
        feats = [get_visit_feature(self.fb[vid], v['start'][1], v['stop'][1], dim=self.fb_dim) for v in visits]

        if _labels is not None:
            labels = []
            for l in _labels:
                labels.append([])
                if self.add_verb:
                    labels[-1].append(l[0])
                if self.add_noun:
                    labels[-1].append(l[1])
        else:
            labels = None

        feats, labels, length = pad_seq_feats(feats, labels, self.max_length, dim=self.fb_dim, append_zero=False)
        
        # print(feats.shape)
        # print(labels.shape, labels.dtype)

        data = dict(
            feature=DC(to_tensor(feats), stack=True, pad_dims=None),
            length=DC(to_tensor(length), stack=True, pad_dims=None),
            num_modalities=DC(to_tensor(1)),
            img_meta=DC(dict(), cpu_only=True),
        )
        if labels is not None:
            data.update(dict(
               gt_label=DC(to_tensor(labels), stack=True, pad_dims=None) 
            ))

        return data


# class EpicNodeDatasetGFB(EpicFeatureDataset):
#     '''
#     Genereate node features (using LSTM model) for GFB
#     '''
#     def __init__(self, graph_root, fb, fb_dim=2048, max_length=8, infer_only=False, **kwargs):
#         super(EpicNodeDatasetGFB, self).__init__(**kwargs)
#         self.fb = mmcv.load(fb)
#         self.fb_dim = fb_dim
#         self.infer_only = infer_only
#         self.max_length = max_length

#         videos = self.fb.keys()
#         self.graphs = epic_utils.load_graphs(graph_root, videos)

#     def __getitem__(self, idx):
#         record = self.video_infos[idx]
      
#         self.cur_graph = epic_utils.get_graph(self.graphs[record.path], record.end_frame)
    
#         feats, length = self.get_cur_node_feats(self.cur_graph, self.fb[record.path])

#         data = dict(
#             feature=DC(to_tensor(feats), stack=True, pad_dims=None),
#             length=DC(to_tensor(length), stack=True, pad_dims=None),
#             num_modalities=DC(to_tensor(1)),
#             img_meta=DC(dict(), cpu_only=True),
#         )

#         return data
    
#     def get_cur_node_feats(self, graph, trunk_features):
#         cur_node = epic_utils.find_last_visit_node(graph)
    
#         visits = graph.node[cur_node]['members']
#         visits = visits[-self.max_length-1:] if not self.infer_only else visits[-self.max_length:]
#         feats = [get_visit_feature(trunk_features, v['start'][1], v['stop'][1], self.fb_dim) for v in visits]
        
#         feats, length = pad_seq_feats(feats, self.max_length, dim=self.fb_dim, append_zero=self.infer_only)
        
#         return feats, length


class EpicNodeDatasetGFB(Dataset):
    '''
    Genereate node features (using LSTM model) for GFB
    '''
    def __init__(self, graph_root, fb, fb_dim=2048, max_length=8, infer_only=False, **kwargs):
        self.dataset = EpicFeatureDataset(**kwargs)
        self.fb = mmcv.load(fb)
        self.fb_dim = fb_dim
        self.infer_only = infer_only
        self.max_length = max_length

        videos = self.fb.keys()
        self.graphs = epic_utils.load_graphs(graph_root, videos)

        self.records = self.build_dataset()

        print("{} records in total".format(len(self.records)))
    
    def build_dataset(self):
        records = set()
        for r in self.dataset.video_infos:
            cur_graph = epic_utils.get_graph(self.graphs[r.path], r.end_frame)
            # next_graph = epic_utils.get_graph(self.graphs[r.path], r.clip_end_frame)

            for g, t in zip([cur_graph], [r.end_frame]):
                for node in sorted(list(g.nodes())):
                    records.add((r.path, t, node))
        
        return list(records)
    
    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        v_path, t, node = self.records[idx]
      
        self.cur_graph = epic_utils.get_graph(self.graphs[v_path], t)
    
        feats, length = self.get_node_feats(self.cur_graph, self.fb[v_path], node)

        data = dict(
            feature=DC(to_tensor(feats), stack=True, pad_dims=None),
            length=DC(to_tensor(length), stack=True, pad_dims=None),
            num_modalities=DC(to_tensor(1)),
            img_meta=DC(dict(), cpu_only=True),
        )

        return data
    
    def get_node_feats(self, graph, trunk_features, node):
        visits = graph.node[node]['members']
        visits = visits[-self.max_length-1:] if not self.infer_only else visits[-self.max_length:]
        feats = [get_visit_feature(trunk_features, v['start'][1], v['stop'][1], self.fb_dim) for v in visits]
        
        feats, _, length = pad_seq_feats(feats, None, self.max_length, dim=self.fb_dim, append_zero=self.infer_only)
        
        return feats, length
