# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

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
import anticipation.datasets.epic_utils as epic_utils


class EpicFutureLabels(Dataset):
    def __init__(self, ann_file, label, test_mode, task, dset, train_timestamps, val_timestamps, train_many_shot=False, **kwargs):
        self.test_mode = test_mode
        self.ann_file = ann_file
        self.label = label
        self.task = task
        self.dset = dset
        self.train_many_shot = train_many_shot
        self.train_timestamps = train_timestamps
        self.val_timestamps = val_timestamps

        self.fps = 60 if self.dset=='epic' else 24

        if self.dset == 'epic':
            manyshot_verbs = sorted(epic_utils.get_many_shot("data/epic/annotations/EPIC_many_shot_verbs.csv"))
            manyshot_nouns = sorted(epic_utils.get_many_shot("data/epic/annotations/EPIC_many_shot_nouns.csv"))
            if train_many_shot:
                self.num_verbs, self.num_nouns, self.num_actions = len(manyshot_verbs), len(manyshot_nouns), 250
            else:
                self.num_verbs, self.num_nouns, self.num_actions = 125, 352, 250
            self.manyshot_verbs, self.manyshot_nouns = manyshot_verbs, manyshot_nouns
        elif self.dset=='gtea':
            manyshot_verbs = sorted(epic_utils.get_many_shot("data/gtea/annotations/gtea_many_shot_verbs.csv"))
            manyshot_nouns = sorted(epic_utils.get_many_shot("data/gtea/annotations/gtea_many_shot_nouns.csv"))
            if train_many_shot:
                self.num_verbs, self.num_nouns, self.num_actions = len(manyshot_verbs), len(manyshot_nouns), 106
            else:
                self.num_verbs, self.num_nouns, self.num_actions = 19, 53, 106
            # self.num_verbs, self.num_nouns, self.num_actions = 19, 53, 106

        # find the most frequent 250 actions we're interested in predicting
        records = [EpicRawFramesRecord(x.strip().split('\t')) for x in open(self.ann_file)]
        int_counts = [(record.label[0], record.label[1]) for record in records]
        int_counts = collections.Counter(int_counts).items()
        int_counts = sorted(int_counts, key=lambda x: -x[1])[0:self.num_actions]
        self.int_to_idx = {interact:idx for idx, (interact, count) in enumerate(int_counts)}

        if task=='anticipation':
            self.data = self.load_annotations_anticipation(ann_file)
        elif task=='recognition':
            self.data = self.load_annotations_recognition(ann_file)

        if train_many_shot:
            for record in self.data:
                record.verbs = [manyshot_verbs.index(x) for x in record.verbs if x in manyshot_verbs]
                record.nouns = [manyshot_nouns.index(x) for x in record.nouns if x in manyshot_nouns]
        
        self.flag = np.ones(len(self), dtype=np.uint8) # just for mmaction sampler interface

        # Only a few nouns/ints will actually have gt positives
        # Pass these as part of the batch to evaluate mAP
        # Don't know how to pass these in the config
        eval_ints = set()
        for record in self.data:
            eval_ints |= set(record.ints)
        eval_set = torch.zeros(1, self.num_actions)
        eval_set[0, list(eval_ints)] = 1
        self.eval_ints = eval_set.byte()

        eval_nouns = set()
        for record in self.data:
            eval_nouns |= set(record.nouns)
        if train_many_shot:# or self.dset != 'epic':
            eval_set = torch.zeros(1, self.num_nouns)
            eval_set[0, list(eval_nouns)] = 1
        else:
            eval_set = torch.zeros(3, self.num_nouns)
            eval_set[0, list(eval_nouns)] = 1
            manyshot = eval_nouns & set(manyshot_nouns)
            rareshot = eval_nouns - set(manyshot_nouns)
            eval_set[1, list(manyshot)] = 1
            eval_set[2, list(rareshot)] = 1
        self.eval_nouns = eval_set.byte()

        eval_verbs = set()
        for record in self.data:
            eval_verbs |= set(record.verbs)
        if train_many_shot:# or self.dset != 'epic':
            eval_set = torch.zeros(1, self.num_verbs)
            eval_set[0, list(eval_verbs)] = 1
        else:
            eval_set = torch.zeros(3, self.num_verbs)
            eval_set[0, list(eval_verbs)] = 1
            manyshot = eval_verbs & set(manyshot_verbs)
            rareshot = eval_verbs - set(manyshot_verbs)
            eval_set[1, list(manyshot)] = 1
            eval_set[2, list(rareshot)] = 1
        self.eval_verbs = eval_set.byte()

    def load_annotations_recognition(self, ann_file):
        vid_lengths = open(self.ann_file.replace('.csv', '_nframes.csv')).read().strip().split('\n')
        vid_lengths = [line.split('\t') for line in vid_lengths]
        vid_lengths = {k:int(v) for k,v in vid_lengths}
       
        records = [EpicRawFramesRecord(x.strip().split('\t')) for x in open(ann_file)]
        records_by_vid = collections.defaultdict(list)
        for record in records:
            record.uid = '%s_%s_%s'%(record.path, record.start_frame, record.end_frame)
            records_by_vid[record.path].append(record)

        records = []
        for vid in records_by_vid:
            vrecords = sorted(records_by_vid[vid], key=lambda record: record.end_frame)
            length = vid_lengths[vid]

            if self.test_mode:
                timestamps = self.val_timestamps # 0.5, 0.75, 1.0]
            else:
                timestamps = self.train_timestamps # 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

            timestamps = [int(frac*length) for frac in timestamps]
            for i, t in enumerate(timestamps):
                past_records = [record for record in vrecords if record.end_frame<=t]
                if len(past_records)<3:
                    continue

                record = EpicRawFramesRecord([vid, 0, t, -1, -1])
                record.verbs = sorted(set([record.label[0] for record in past_records]))
                record.nouns = sorted(set([record.label[1] for record in past_records]))
                record.ints = sorted(set([self.int_to_idx[(record.label[0], record.label[1])] for record in past_records if (record.label[0], record.label[1]) in self.int_to_idx]))
                record.ratio_idx = i
                records.append(record)
                
        print ('(REC) Collected %d records'%len(records))

        return records

    def load_annotations_anticipation(self, ann_file):
        vid_lengths = open(self.ann_file.replace('.csv', '_nframes.csv')).read().strip().split('\n')
        vid_lengths = [line.split('\t') for line in vid_lengths]
        vid_lengths = {k:int(v) for k,v in vid_lengths}

        records = [EpicRawFramesRecord(x.strip().split('\t')) for x in open(ann_file)]

        records_by_vid = collections.defaultdict(list)
        for record in records:
            record.uid = '%s_%s_%s'%(record.path, record.start_frame, record.end_frame)
            records_by_vid[record.path].append(record)

        records = []
        for vid in records_by_vid:
            vrecords = sorted(records_by_vid[vid], key=lambda record: record.end_frame)
            length = vid_lengths[vid]

            if self.test_mode:
                timestamps = self.val_timestamps
            else:
                timestamps = self.train_timestamps                           # [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        
            timestamps = [int(frac*length) for frac in timestamps]
            for i, t in enumerate(timestamps):
                past_records = [record for record in vrecords if record.end_frame<=t]
                future_records = [record for record in vrecords if record.start_frame>t]
                if len(past_records)<3 or len(future_records)<3:
                    continue

                record = EpicRawFramesRecord([vid, 0, t, -1, -1])
                record.verbs = sorted(set([record.label[0] for record in future_records]))
                record.nouns = sorted(set([record.label[1] for record in future_records]))
                record.ints = sorted(set([self.int_to_idx[(record.label[0], record.label[1])] for record in future_records if (record.label[0], record.label[1]) in self.int_to_idx]))
                record.ratio_idx = i
                records.append(record)
                
        print ('(ANT) Collected %d records'%len(records))

        return records


    def get_ann_info(self, idx):
        return {
            'path': self.data[idx].path,
            'num_frames': self.data[idx].num_frames,
            'label': self.data[idx].label
        }
    
        
    def __getitem__(self, idx):
        record = self.data[idx]
        data = dict(
            num_modalities=DC(to_tensor(1)),
            img_meta=DC(dict(), cpu_only=True),
            gt_label=DC(to_tensor(0)), # Dummy
            feature=DC(to_tensor(torch.zeros(2048))), # Dummy
            ratio_idx=DC(to_tensor(record.ratio_idx), stack=True, pad_dims=None)
        )

        if self.label=='int':
            ints = torch.zeros(self.num_actions)
            ints[record.ints] = 1
            data.update(dict(
                labels=DC(to_tensor(ints), stack=True, pad_dims=None),
                label_mask=DC(to_tensor(self.eval_ints), stack=True, pad_dims=None),
            ))
        elif self.label=='noun':
            nouns = torch.zeros(self.num_nouns)
            nouns[record.nouns] = 1
            data.update(dict(
                labels=DC(to_tensor(nouns), stack=True, pad_dims=None),
                label_mask=DC(to_tensor(self.eval_nouns), stack=True, pad_dims=None),
            ))
        elif self.label=='verb':
            verbs = torch.zeros(self.num_verbs)
            verbs[record.verbs] = 1
            data.update(dict(
                labels=DC(to_tensor(verbs), stack=True, pad_dims=None),
                label_mask=DC(to_tensor(self.eval_verbs), stack=True, pad_dims=None),
            ))
      
        return data

    def __len__(self):
        return len(self.data)


class EpicFutureLabelsI3D(EpicFutureLabels):
    def __init__(self, ann_file, label, test_mode, task, dset, fb, lfb_window=64, FPS=60, fb_dim=2048, **kwargs):
        super().__init__(ann_file, label, test_mode, task, dset, **kwargs)
        self.fb = mmcv.load(fb)
        self.lfb_window = lfb_window
        self.FPS = FPS
        self.fb_dim = fb_dim

        print("{} {} samples. Traintime {} and valtime {}".format(task, len(self.data), self.train_timestamps, self.val_timestamps))

    def sample_lfb(self, start_idx, end_idx, video_lfb):
        candidate_idx = []
        for frame_idx in range(start_idx+31, end_idx-31):
            if frame_idx in video_lfb.keys():
                candidate_idx.append(frame_idx)

        # Uniformly select self.lfb_window features from this set
        candidate_idx = [candidate_idx[ix] for ix in np.round(np.linspace(0, len(candidate_idx) - 1, self.lfb_window)).astype(int)]

        out_lfb = [video_lfb[idx] for idx in candidate_idx]
        out_lfb = np.array(out_lfb)
        return out_lfb.astype(np.float32)


    def __getitem__(self, idx):
        record = self.data[idx]
        data = super().__getitem__(idx)
        lfb = self.sample_lfb(record.start_frame, record.end_frame, self.fb[record.path])
        data.update({'lfb':DC(to_tensor(lfb), stack=True, pad_dims=None)})

        return data

class EpicFutureLabelsGFB(EpicFutureLabelsI3D):

    def __init__(self, ann_file, label, test_mode, task, dset, fb, graph_root, **kwargs):
        super().__init__(ann_file, label, test_mode, task, dset, fb, **kwargs)

        timestamps = self.val_timestamps if self.test_mode else self.train_timestamps
        data_fl = 'data/%s/future_labels/gfb_data_%s_%s_%s.pth'%(
            dset,
            osp.basename(self.ann_file),
            task,
            ",".join([str(x) for x in timestamps])
        )
        if self.train_many_shot:
            data_fl = data_fl.replace(".pth", "_manyshot.pth")
        if not osp.exists(data_fl):
            os.makedirs('data/%s/future_labels'%(dset), exist_ok=True)
            videos = self.fb.keys()
            self.parse_graph_data(graph_root, videos, data_fl)
        data = torch.load(data_fl)['records']
        assert len(data) == len(self.data)
        self.data = data

        self.node_num_member = 8

        print("{} {} samples. Traintime {} and valtime {}".format(task, len(self.data), self.train_timestamps, self.val_timestamps))

    def visits_to_labels(self, visits):

        if len(visits)==0:
            return torch.zeros(self.num_verbs)-1, torch.zeros(self.num_nouns)-1, torch.zeros(self.num_actions)-1

        v_id = visits[0]['start'][0]

        n_frames = []
        for visit in visits:
            n_frames += [(v_id, f_id) for f_id in range(visit['start'][1], visit['stop'][1]+1)]
        records = [self.frame_to_record[frame] for frame in n_frames if frame in self.frame_to_record]
        records = {record.uid:record for record in records}.values() # remove duplicate entries

        def get_dist(recs, N, label_fn, manyshot=None):
            counts = []
            for record in recs:
                counts.append(label_fn(record))
            if manyshot is not None:
                counts = [manyshot.index(x) for x in counts if x in manyshot]
            counts = collections.Counter(counts)

            dist = torch.zeros(N)
            if len(counts)>0:
                for item, count in counts.items():
                    dist[item] = 1 # not count
                # dist = dist/dist.sum()
            return dist

        verb_dist = get_dist(records, self.num_verbs, lambda record: record.label[0], self.manyshot_verbs if self.train_many_shot else None)
        noun_dist = get_dist(records, self.num_nouns, lambda record: record.label[1], self.manyshot_nouns if self.train_many_shot else None)
        int_dist = get_dist([record for record in records if (record.label[0], record.label[1]) in self.int_to_idx], self.num_actions, lambda record: self.int_to_idx[(record.label[0], record.label[1])])

        return verb_dist, noun_dist, int_dist

    def get_future_labels(self, graph, end_frame, graph_data):

        frames, graphs = graph_data['frames'], graph_data['graphs']
        final_graph = graphs[-1]['G']

        # for each node, get future labels
        future_verbs, future_labels, future_ints = [], [], []
        for node in sorted(graph.nodes()):
            visits = copy.deepcopy(final_graph.node[node]['members'])
            for visit in visits:
                visit['start'] = frames[visit['start']]
                visit['stop'] = frames[visit['stop']]

            if self.task=='anticipation':
                future_visits = [visit for visit in visits if visit['start'][1]>end_frame]
            elif self.task=='recognition':
                future_visits = [visit for visit in visits if visit['stop'][1]<end_frame]

            vfuture, nfuture, ifuture = self.visits_to_labels(future_visits)
            future_verbs.append(vfuture)
            future_labels.append(nfuture)
            future_ints.append(ifuture)

        future_verbs = torch.stack(future_verbs, 0)
        future_labels = torch.stack(future_labels, 0)
        future_ints = torch.stack(future_ints, 0)

        return {'verbs':future_verbs, 'nouns':future_labels, 'ints':future_ints}


    def parse_graph_data(self, graph_root, videos, out_fl):

        if self.dset=='epic':
            record_to_vid = lambda record: record.path.split('/')[1]
        elif self.dset=='gtea':
            record_to_vid = lambda record: record.path

        records = [EpicRawFramesRecord(x.strip().split('\t')) for x in open(self.ann_file)]
        frame_to_record = {}
        for record in records:
            v_id = record_to_vid(record)
            record.uid = '%s_%s_%s'%(v_id, record.start_frame, record.end_frame)
            for f_id in range(record.start_frame, record.end_frame+1):
                frame_to_record[(v_id, f_id)] = record
        self.frame_to_record = frame_to_record


        # ALL graphs do not fit in memory, and the data is sequential anyway
        @lru_cache(maxsize=3)
        def get_graph_data(record):
            vid = record_to_vid(record)
            return torch.load("{}/{}_graph.pth".format(graph_root, vid))
      
        for record in tqdm.tqdm(self.data, total=len(self.data)):
            graph_data = get_graph_data(record)
            record.graph = epic_utils.get_graph(graph_data, record.end_frame)
            record.future_labels = self.get_future_labels(record.graph, record.end_frame, graph_data)
            
        torch.save({'records':self.data}, out_fl)

    def get_visit_feature(self, trunk_features, start, stop, dim):
        inds = []
        for fid in range(start - 31, stop - 31):
            if fid in trunk_features.keys():
                inds.append(fid)

        if len(inds)==0:
            return None

        if self.test_mode:
            f_id = inds[len(inds)//2]
        else:
            f_id = inds[np.random.randint(len(inds))]

        feat = trunk_features[f_id]
        feat = to_tensor(feat)
        return feat

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
            feats = [feats[idx] for idx in np.round(np.linspace(0, len(feats) - 1, self.node_num_member)).astype(int)]
            length = len(feats)

            feats = torch.stack(feats, 0) # (K, 2048)
            node_feats.append(feats)
            node_length.append(length)
            
        node_feats = torch.stack(node_feats, 0)  # (N, K, 2048)
        node_length = torch.LongTensor(node_length)

        return node_feats, node_length
    
    def process_graph_feats(self, graph, trunk_features, future_labels):
        # Drop useless visits (VERY IMPORTANT FOR GTEA's BLACK FRAMES !!!!!)
        keep = torch.ones((len(graph.nodes())))
        node_feats, node_length = self.get_node_feats(graph, trunk_features)
        if self.dset=='gtea':
            for i, node in enumerate(sorted(graph.nodes())):
                visits = graph.node[node]['members']
                if len(visits)==1 and visits[0]['stop'][1]-visits[0]['start'][1]<self.fps:
                    graph.remove_node(node)
                    keep[i] = 0

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

    def __getitem__(self, idx):
        record = self.data[idx]
        data = super().__getitem__(idx)

        g = self.process_graph_feats(record.graph, self.fb[record.path], future_labels=record.future_labels)
        data.update({'gfb':DC(g, stack=False, cpu_only=True)})

        return data


class EpicFutureLabelsGFB1(EpicFutureLabelsGFB):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_visit_feature(self, trunk_features, start, stop, dim):
        inds = []
        for fid in range(start - 31, stop - 31):
            if fid in trunk_features.keys():
                inds.append(fid)

        if len(inds)==0:
            return None

        if self.test_mode:
            f_id = inds[len(inds)//2]
            feats = [to_tensor(trunk_features[fid]) for fid in inds]
        
            feat = torch.stack(feats, 0).mean(0, keepdim=False)
        else:
            f_id = inds[np.random.randint(len(inds))]

            feat = trunk_features[f_id]
            feat = to_tensor(feat)
        
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