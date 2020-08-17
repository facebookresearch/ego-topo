import bisect
import copy
import os.path as osp
import random
from functools import partial

import dgl
import numpy as np
import torch
from torch.utils.data import Dataset

import mmcv
from mmcv.parallel import DataContainer as DC
from mmcv.parallel import collate

from .epic_utils import EpicRawFramesRecord, to_tensor, get_visit_feature
import epic.datasets.epic_utils as epic_utils


class EpicFeatureDataset(Dataset):
    def __init__(self,
                 ann_file,
                 feature_file,
                 test_mode=True,
                 anticipation_task=False,
                 auxiliary_feature=None,
                 anti2=False,
                 N=1,
                 anti_ta=60,
                 anti_to=64):

        self.clip_trunk_features = mmcv.load(feature_file)
        self.anticipation_task = anticipation_task
        self.anti2 = anti2
        self.N = N
        self.anti_ta = anti_ta
        self.anti_to = anti_to
    
        self.auxiliary_feature = mmcv.load(auxiliary_feature) if auxiliary_feature is not None else None
        # load annotations
        self.video_infos = self.load_annotations(ann_file)
        # self.video_infos = self.video_infos[:32]

        self.test_mode = test_mode
        # just for mmaction sampler interface
        self.flag = np.ones(len(self), dtype=np.uint8)

    def __len__(self):
        return len(self.video_infos)

    def load_annotations(self, ann_file):
        records = [EpicRawFramesRecord(x.strip().split('\t')) for x in open(ann_file)]
        if self.anticipation_task:
            if not self.anti2:
                records = self.convert_to_anticipation(records)
            else:
                records = self.convert_to_anticipation2(records)
        return records

    def convert_to_anticipation(self, records):
        print("converting to dataset for anticipation task!")
        anti_records = []
        removed = 0
        for rec in records:
            if rec.start_frame <= self.anti_ta:
                removed += 1
                continue

            start = max(1, rec.start_frame - self.anti_ta - self.anti_to)
            end = rec.start_frame - self.anti_ta
            anti_records.append(
                EpicRawFramesRecord(
                    [rec.path, start, end] + rec.label,
                    [rec.start_frame, rec.end_frame]
                ),
            )

        print("removed {} clip".format(removed))
        return anti_records
    
    def remove_consecutive_labels(self, records):
        merged_records = []
        last_label = None
        merged = 0
        for rec in records:
            current_label = (rec.path, ) + tuple(rec.label)
            if current_label == last_label:
                last_rec = merged_records.pop(-1)
                merged_rec = EpicRawFramesRecord(
                    [rec.path, last_rec.start_frame, rec.end_frame] + rec.label, 
                    [rec.start_frame, rec.end_frame]
                )
                merged_records.append(merged_rec)
                merged += 1
                continue
            merged_records.append(rec)
            last_label = current_label

        print("merged {} consecutive clips".format(merged))
        return merged_records 

    # records is already sorted
    def convert_to_anticipation2(self, records):

        print("converting to dataset for anticipation task (clip level)!")
        records = self.remove_consecutive_labels(records)
        anti_records = []
        removed = 0

        for idx in range(len(records)-self.N):
            curr_rec = records[idx]
            next_rec = records[idx+self.N]

            if next_rec.start_frame - curr_rec.end_frame <= self.anti_ta or next_rec.path != curr_rec.path:
                removed += 1
                continue

            # Label is of NEXT interaction
            anti_records.append(
                EpicRawFramesRecord(
                    [curr_rec.path, curr_rec.start_frame, curr_rec.end_frame] + next_rec.label, 
                    [next_rec.start_frame, next_rec.end_frame]
                ),
            )

        print("removed {} clip".format(removed))
        return anti_records

    def get_ann_info(self, idx):
        return {
            'path': self.video_infos[idx].path,
            'num_frames': self.video_infos[idx].num_frames,
            'label': self.video_infos[idx].label
        }
    
    def get_clip_frames(self, frame_ids, start_idx, end_idx):
        sorted_ids = sorted(frame_ids)
        idx = bisect.bisect(sorted_ids, end_idx - 31)
        ret = [sorted_ids[idx - 1]]
        for i in range(idx - 2, 0):
            if sorted_ids[i] < start_idx + 31:
                break
            ret.append(sorted_ids[i])
        
        return ret

    def sample_index(self, frame_ids, test_mode):
        assert len(frame_ids) > 0

        if test_mode == True:
            idx = len(frame_ids) // 2
        else:
            idx = random.randint(0, len(frame_ids) - 1)
        
        return frame_ids[idx]
        
    def __getitem__(self, idx):
        record = self.video_infos[idx]
        features = self.clip_trunk_features[record.path]
        frame_ids = self.get_clip_frames(features.keys(), record.start_frame, record.end_frame)
        frame_id = self.sample_index(frame_ids, self.test_mode)
        feature = features[frame_id]

        data = dict(
            gt_label=DC(to_tensor(record.label), stack=True, pad_dims=None),
            feature=DC(to_tensor(feature), stack=True, pad_dims=None),
            num_modalities=DC(to_tensor(1)),
            img_meta=DC(dict(), cpu_only=True),
        )
      
        return data


# # Alternate anticipation task. 
# class EpicFeatureDatasetAnt2(EpicFeatureDataset):
#     def __init__(self,
#                  ann_file,
#                  feature_file,
#                  test_mode=True,
#                  anticipation_task=True,
#                  N=1):

#         self.N = N # Number of clips to look ahead
#         super().__init__(ann_file, feature_file, test_mode, anticipation_task, 60, 64)
        

#     def remove_consecutive_labels(self, records):
#         merged_records = []
#         last_label = None
#         merged = 0
#         for rec in records:
#             current_label = (rec.path, ) + tuple(rec.label)
#             if current_label == last_label:
#                 last_rec = merged_records.pop(-1)
#                 merged_rec = EpicRawFramesRecord(
#                     [rec.path, last_rec.start_frame, rec.end_frame] + rec.label, 
#                     [rec.start_frame, rec.end_frame]
#                 )
#                 merged_records.append(merged_rec)
#                 merged += 1
#                 continue
#             merged_records.append(rec)
#             last_label = current_label

#         print("merged {} consecutive clips".format(merged))
#         return merged_records 

#     # records is already sorted
#     def convert_to_anticipation(self, records):

#         print("converting to dataset for anticipation task (clip level)!")
#         records = self.remove_consecutive_labels(records)
#         anti_records = []
#         removed = 0

#         for idx in range(len(records)-self.N):
#             curr_rec = records[idx]
#             next_rec = records[idx+self.N]

#             if next_rec.start_frame - curr_rec.end_frame <= self.anti_ta or next_rec.path != curr_rec.path:
#                 removed += 1
#                 continue

#             # Label is of NEXT interaction
#             anti_records.append(
#                 EpicRawFramesRecord(
#                     [curr_rec.path, curr_rec.start_frame, curr_rec.end_frame] + next_rec.label, 
#                     [curr_rec.start_frame, curr_rec.end_frame]
#                 ),
#             )

#         print("removed {} clip".format(removed))
#         return anti_records


class EpicFeatureDatasetLFB(EpicFeatureDataset):
    """
    Add LFB features.
    """
    def __init__(self,
                 lfb,
                 lfb_window=40,
                 lfb_window_mode='center',
                 FPS=60,
                 lfb_dim=2048,
                 **kwargs):
        super(EpicFeatureDatasetLFB, self).__init__(**kwargs)

        self.lfb = mmcv.load(lfb)
        self.lfb_window = lfb_window
        self.lfb_window_mode = lfb_window_mode
        self.FPS = FPS
        self.lfb_dim = lfb_dim
        
        assert lfb_window_mode in ['center', 'end']

    def __getitem__(self, idx):
        record = self.video_infos[idx]
        data = super(EpicFeatureDatasetLFB, self).__getitem__(idx)

        lfb = self.sample_lfb(
            record.start_frame,
            record.end_frame,
            self.lfb[record.path],
        )
        data.update({'lfb':DC(to_tensor(lfb), stack=True, pad_dims=None)})

        return data

    def sample_lfb(self, start_idx, end_idx, video_lfb):
        half_len = (self.lfb_window + 1) * self.FPS // 2

        if self.lfb_window_mode == 'center':
            center_idx = (start_idx + end_idx) // 2
            lower = center_idx - half_len
            upper = center_idx + half_len
        else:
            lower = end_idx - half_len * 2
            upper = end_idx

        out_lfb = []
        # Note clip features has 1s window size
        for frame_idx in range(upper - 31, lower + 31, -1):
            if frame_idx in video_lfb.keys():
                if len(out_lfb) < self.lfb_window:
                    out_lfb.append(video_lfb[frame_idx])
                else:
                    break

        out_lfb = np.array(out_lfb)
        if out_lfb.shape[0] < self.lfb_window:
            new_out_lfb = np.zeros((self.lfb_window, self.lfb_dim))
            if out_lfb.shape[0] > 0:
                new_out_lfb[-out_lfb.shape[0]:] = out_lfb
            out_lfb = new_out_lfb

        return out_lfb.astype(np.float32)


class EpicFeatureDatasetGFB(EpicFeatureDataset):
    '''
    Use precomputed generated graphs in history for each video
    '''
    def __init__(self, graph_root, fb, fb_dim=2048, add_next_node=False, node_feat_mode='avg', node_num_member=3, **kwargs):
        super(EpicFeatureDatasetGFB, self).__init__(**kwargs)
        self.fb = mmcv.load(fb)
        self.fb_dim = fb_dim

        videos = self.fb.keys()
        # videos = sorted(list(videos))[:5]
        # print(videos)
        self.graphs = epic_utils.load_graphs(graph_root, videos)

        assert node_feat_mode in ["last", "avg", "max"]
        self.node_feat_mode = node_feat_mode
        self.node_num_member = node_num_member
        self.add_next_node = add_next_node

    def __getitem__(self, idx):
        record = self.video_infos[idx]
        data = super(EpicFeatureDatasetGFB, self).__getitem__(idx)

        self.cur_graph = epic_utils.get_graph(self.graphs[record.path], record.end_frame)
        self.next_graph = epic_utils.get_graph(self.graphs[record.path], record.clip_end_frame) if self.add_next_node else None
        node_feats = self.get_node_feats(self.cur_graph, self.fb[record.path])
        g = self.process_graph_feats(self.cur_graph, node_feats, next_graph=self.next_graph)

        data.update({'gfb':DC(g, stack=False, cpu_only=True)})

        return data
    
    def get_node_feats(self, graph, trunk_features):
        node_feats = []
        for node in sorted(list(graph.nodes())):
            visits = graph.node[node]['members']
            if self.node_num_member > 0:
                visits = visits[-self.node_num_member:]
            if self.node_feat_mode == 'last':
                visits = visits[-1:]

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
    
    def process_graph_feats(self, graph, node_feats, next_graph=None):
        # -------------------------------------------------------------------#
        # Make the dgl graph now
        nodes = sorted(list(graph.nodes()))
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

        g.ndata['feats'] = node_feats

        cur_status = torch.zeros(len(nodes))
        # cur_node = 0 if graph.last_state['node'] is None else node_to_idx[graph.last_state['node']]
        cur_node = epic_utils.find_last_visit_node(graph)
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


class EpicFeatureDatasetGFBNode(EpicFeatureDatasetGFB):
    '''
    Use precomputed generated graphs in history for each video
    '''
    def __init__(self, node_fb, keep_ori_gfb=False, **kwargs):
        super(EpicFeatureDatasetGFBNode, self).__init__(**kwargs)
       
        self.node_fb = mmcv.load(node_fb)
        self.keep_ori_gfb = keep_ori_gfb

    def __getitem__(self, idx):
        record = self.video_infos[idx]
        # print(record.path, record.end_frame)
        data = super(EpicFeatureDatasetGFB, self).__getitem__(idx)

        self.cur_graph = epic_utils.get_graph(self.graphs[record.path], record.end_frame)
        self.next_graph = epic_utils.get_graph(self.graphs[record.path], record.clip_end_frame) if self.add_next_node else None
        node_feats = self.get_aux_node_feats(self.cur_graph, self.node_fb[(record.path, record.end_frame)])
        if self.keep_ori_gfb:
            ori_node_feats = self.get_node_feats(self.cur_graph, self.fb[record.path])
            node_feats = torch.cat((node_feats, ori_node_feats), dim=1)
        g = self.process_graph_feats(self.cur_graph, node_feats, next_graph=self.next_graph)

        data.update({'gfb':DC(g, stack=False, cpu_only=True)})

        return data
    
    def get_aux_node_feats(self, graph, trunk_features):
        node_feats = []
        # print(trunk_features.keys(), graph.nodes())
        for node in sorted(list(graph.nodes())):
            feats = trunk_features[node]

            node_feats.append(to_tensor(feats))

        node_feats = torch.stack(node_feats, 0)  # (N, 2048)

        return node_feats


class EpicFeatureDatasetGFBLFB(EpicFeatureDatasetGFB):
    """
    Add GFB + LFB features.
    """
    def __init__(self,
                 lfb,
                 lfb_window=40,
                 lfb_window_mode='center',
                 FPS=60,
                 lfb_dim=2048,
                 **kwargs):
        super(EpicFeatureDatasetGFBLFB, self).__init__(**kwargs)

        self.lfb = mmcv.load(lfb)
        self.lfb_window = lfb_window
        self.lfb_window_mode = lfb_window_mode
        self.FPS = FPS
        self.lfb_dim = lfb_dim
        
        assert lfb_window_mode in ['center', 'end']

    def __getitem__(self, idx):
        record = self.video_infos[idx]
        data = super(EpicFeatureDatasetGFBLFB, self).__getitem__(idx)

        lfb = self.sample_lfb(
            record.start_frame,
            record.end_frame,
            self.lfb[record.path],
        )
        data.update({'lfb':DC(to_tensor(lfb), stack=True, pad_dims=None)})

        return data

    def sample_lfb(self, start_idx, end_idx, video_lfb):
        half_len = (self.lfb_window + 1) * self.FPS // 2

        if self.lfb_window_mode == 'center':
            center_idx = (start_idx + end_idx) // 2
            lower = center_idx - half_len
            upper = center_idx + half_len
        else:
            lower = end_idx - half_len * 2
            upper = end_idx

        out_lfb = []
        # Note clip features has 1s window size
        for frame_idx in range(upper - 31, lower + 31, -1):
            if frame_idx in video_lfb.keys():
                if len(out_lfb) < self.lfb_window:
                    out_lfb.append(video_lfb[frame_idx])
                else:
                    break

        out_lfb = np.array(out_lfb)
        if out_lfb.shape[0] < self.lfb_window:
            new_out_lfb = np.zeros((self.lfb_window, self.lfb_dim))
            if out_lfb.shape[0] > 0:
                new_out_lfb[-out_lfb.shape[0]:] = out_lfb
            out_lfb = new_out_lfb

        return out_lfb.astype(np.float32)    

if __name__ == '__main__':
    split="S1"
    dataset = EpicFeatureDataset(
        ann_file='data/epic/split/train_{}.csv'.format(split),
        feature_file='data/features/train_lfb_{}.pkl'.format(split),
        test_mode=False,
        anticipation_task=True
    )
    imgs_per_gpu=128

    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=256,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=partial(collate, samples_per_gpu=imgs_per_gpu)
    )
    for batch in loader:
        print(batch["feature"].data[0].shape)
        print(batch["gt_label"].data[0].shape)

        break
