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
import tqdm
import collections

import mmcv
from mmcv.parallel import DataContainer as DC

from .epic_utils import EpicRawFramesRecord, to_tensor
import epic.datasets.epic_utils as epic_utils
from .epic_features_dataset import EpicFeatureDataset


def get_visit_feature(trunk_features, start, stop, dim, test_mode):
    feats = []
    for fid in range(start - 31, stop - 31):
        if fid in trunk_features.keys():
            feats.append(to_tensor(trunk_features[fid]))

    if test_mode:
        feats = feats[len(feats)//2]
    else:
        feats = feats[np.random.randint(len(feats))]

    return feats

def has_feature(trunk_features, start, stop):
    for fid in range(start - 31, stop - 31):
        if fid in trunk_features.keys():
            return True
    return False


# def get_visit_feature(trunk_features, start, stop, dim, v_id, test_mode):
#     inds = []
#     for fid in range(start - 31, stop - 31):

#         if fid in trunk_features.keys():
#             inds.append(fid)

#     if len(inds)==0:
#         return None

#     if test_mode:
#         f_id = inds[len(inds)//2]
#     else:
#         f_id = inds[np.random.randint(len(inds))]

#     feat = torch.load('/vision/vision_users/tushar/work/topo-ego/data/epic_frame_features_r152/%s/%s.pth'%(v_id, f_id))
#     feat = to_tensor(feat)

#     return feat

class EpicPretrainDataset(Dataset):
    def __init__(self, graph_root, fb, ann_file=None, fb_dim=2048, max_length=16, test_mode=True):

        self.fb = mmcv.load(fb)
        self.fb_dim = fb_dim
        self.max_length = max_length
        self.ann_file = ann_file
        self.test_mode = test_mode

        uid = osp.basename(ann_file)
        data_fl = 'data/pretrain_data_%s.pth'%uid
        if not osp.exists(data_fl):
            videos = list(self.fb.keys())
            self.build_dataset(graph_root, videos, data_fl)
        self.data = torch.load(data_fl)

        # just for mmaction sampler interface
        self.flag = np.ones(len(self), dtype=np.uint8)

    def visits_to_intdist(self, visits):
        v_id = visits[0]['start'][0]

        n_frames = []
        for visit in visits:
            n_frames += [(v_id, f_id) for f_id in range(visit['start'][1], visit['stop'][1]+1)]

        records = [self.frame_to_record[frame] for frame in n_frames if frame in self.frame_to_record]
        records = {record.uid:record for record in records}.values() # remove duplicate entries

        def get_dist(recs, N, label_fn):
            counts = []
            for record in recs:
                counts.append(label_fn(record))
            counts = collections.Counter(counts)

            dist = torch.zeros(N)
            if len(counts)>0:
                for item, count in counts.items():
                    dist[item] = count
                dist = dist/dist.sum()
            return dist

        verb_dist = get_dist(records, 125, lambda record: record.label[0])
        noun_dist = get_dist(records, 352, lambda record: record.label[1])
        int_dist = get_dist([record for record in records if (record.label[0], record.label[1]) in self.int_to_idx], 250, lambda record: self.int_to_idx[(record.label[0], record.label[1])])

        return verb_dist, noun_dist, int_dist

    def build_dataset(self, graph_root, videos, out_file):

        video_infos = [EpicRawFramesRecord(x.strip().split('\t')) for x in open(self.ann_file)]
        frame_to_record = {}
        int_counts = []
        for record in video_infos:
            record.v_id = record.path.split('/')[1]
            for f_id in range(record.start_frame, record.end_frame+1):
                frame_to_record[(record.v_id, f_id)] = record
            record.uid = '%s_%s_%s'%(record.path, record.start_frame, record.end_frame)
            int_counts.append((record.label[0], record.label[1]))
        self.frame_to_record = frame_to_record

        int_counts = collections.Counter(int_counts).items()
        int_counts = sorted(int_counts, key=lambda x: -x[1])[0:250]
        self.int_to_idx = {interact:idx for idx, (interact, count) in enumerate(int_counts)}

        uid_to_labels = {}
        for record in video_infos:
            uid_to_labels[record.uid] = {'verb':record.label[0], 'noun':record.label[1]}
        self.uid_to_labels = uid_to_labels

        graphs = {}
        for vid in tqdm.tqdm(videos):
            graphs[vid] = torch.load("{}/{}_graph.pth".format(graph_root, vid[4:]))
        print("Finished pre-loading graphs")

        records = []
        for vid, graph_history in graphs.items():
            frames, graphs = graph_history["frames"], graph_history["graphs"]

            G = graphs[-1]['G']
            for node in G.nodes:
                visits = G.node[node]['members']
                visits = [{'start': frames[visit['start']], 'stop': frames[visit['stop']]} for visit in visits]
                visits = [visit for visit in visits if has_feature(self.fb[vid], visit['start'][1], visit['stop'][1])]

                vlabels, nlabels = [], []
                _visits = []
                for visit in visits:
                    v_id = visit['start'][0]
                    uid = '%s/%s_%s_%s'%(v_id.split('_')[0], v_id, visit['start'][1], visit['stop'][1])

                    if uid not in self.uid_to_labels:
                        continue

                    verb, noun = self.uid_to_labels[uid]['verb'], self.uid_to_labels[uid]['noun']
                    _visits.append(visit)
                    vlabels.append(verb)
                    nlabels.append(noun)
                visits = _visits

                if len(visits)<5:
                    continue

                verb_dist, noun_dist, int_dist = self.visits_to_intdist(visits)
                
                for t in range(4, len(visits)-1):
                    start = max(0, t - self.max_length)
                    records.append({'vid':vid, 'node':node, 'visits':visits[start:t], 'src':0,
                                    'verb_dist':verb_dist, 'noun_dist':noun_dist, 'int_dist':int_dist,
                                    'verb_labels':vlabels[start:t], 'noun_labels':nlabels[start:t]})


        # # Add in original video_infos entries
        # for record in video_infos:
        #     visit = {'start':(record.v_id, record.start_frame), 'stop': (record.v_id, record.end_frame)}
        #     if not has_feature(self.fb[record.path], record.start_frame, record.end_frame):
        #         continue

        #     verb_dist, noun_dist, int_dist = self.visits_to_intdist([visit])
        #     records.append({'vid':record.path, 'node':-1, 'visits':[visit], 'verb_dist':verb_dist, 'noun_dist':noun_dist, 'int_dist':int_dist, 'src': 1})

        print("finish building {} samples".format(len(records)))

        torch.save(records, out_file)

    def __getitem__(self, idx):
        record = self.data[idx]
        visits = record['visits']

        feats = [get_visit_feature(self.fb[record['vid']], v['start'][1], v['stop'][1], dim=self.fb_dim, test_mode=self.test_mode) for v in visits]
        # feats = [get_visit_feature(self.fb[v['vid']], v['start'][1], v['stop'][1], v_id=v['vid'].split('/')[-1], dim=self.fb_dim, test_mode=self.test_mode) for v in visits]

        length = len(feats)
        if len(feats)<self.max_length:
            feats = feats + [torch.zeros(2048)]*(self.max_length - length)
        feats = torch.stack(feats, 0)

        verb_labels = record['verb_labels']
        noun_labels = record['noun_labels']

        if len(verb_labels)<self.max_length:
            verb_labels = verb_labels + [0]*(self.max_length - len(verb_labels))
        if len(noun_labels)<self.max_length:
            noun_labels = noun_labels + [0]*(self.max_length - len(noun_labels))

        data = dict(
            feature=DC(to_tensor(feats), stack=True, pad_dims=None),
            length=DC(to_tensor(length), stack=True, pad_dims=None),
            num_modalities=DC(to_tensor(1)),
            img_meta=DC(dict(), cpu_only=True),
            verb_dist=DC(to_tensor(record['verb_dist']), stack=True, pad_dims=None),
            noun_dist=DC(to_tensor(record['noun_dist']), stack=True, pad_dims=None),
            int_dist=DC(to_tensor(record['int_dist']), stack=True, pad_dims=None),
            src=DC(to_tensor(record['src']), stack=True, pad_dims=None),
            verb_labels=DC(to_tensor(verb_labels), stack=True, pad_dims=None),
            noun_labels=DC(to_tensor(noun_labels), stack=True, pad_dims=None),
        )

        return data

    def __len__(self):
        return len(self.data)