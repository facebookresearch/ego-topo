import bisect
import copy
import os.path as osp
import random
from functools import partial

import dgl
import numpy as np
import torch
from torch.utils.data import Dataset
import collections
import itertools
import tqdm
from torchvision import transforms
from PIL import Image
import networkx as nx

import mmcv
from mmcv.parallel import DataContainer as DC


from .epic_utils import EpicRawFramesRecord, to_tensor
import epic.datasets.epic_utils as epic_utils



def default_transform(test_mode):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if not test_mode:
        transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.RandomCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std)
                    ])

    else:
        transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std)
            ])

    return transform


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

        self.test_mode = test_mode
        # just for mmaction sampler interface
        self.flag = np.ones(len(self), dtype=np.uint8)

        self.transform = default_transform(test_mode)


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

        # records = self.remove_consecutive_labels(records)

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
            gt_label=DC(to_tensor(record.label[0]), stack=True, pad_dims=None), # verbs
            feature=DC(to_tensor(feature), stack=True, pad_dims=None),
            num_modalities=DC(to_tensor(1)),
            img_meta=DC(dict(), cpu_only=True),
        )
      
        return data

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


def has_feature(trunk_features, start, stop):
    for fid in range(start - 31, stop - 31):
        if fid in trunk_features.keys():
            return True
    return False


class EpicFeatureDatasetGFB(EpicFeatureDataset):
    '''
    Use precomputed generated graphs in history for each video
    '''
    def __init__(self, graph_root, fb, fb_dim=2048, add_next_node=False, node_feat_mode='avg', node_num_member=3, **kwargs):
        super(EpicFeatureDatasetGFB, self).__init__(**kwargs)
        self.fb = mmcv.load(fb)
        self.fb_dim = fb_dim

        self.verbs = open('data/verbs.txt').read().strip().split('\n')
        self.nouns = open('data/nouns.txt').read().strip().split('\n')

        uid = osp.basename(kwargs['ann_file'])
        data_fl = 'data/gfb_data_%s.pth'%uid
        if not osp.exists(data_fl):
            videos = self.fb.keys()
            self.parse_graph_data(graph_root, videos, data_fl)
        self.video_infos = torch.load(data_fl)['records']

        self.node_num_member = 16
        self.node_feat_mode = 'avg'
        self.add_next_node = True


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

    def get_graph_intdist(self, graph_history):
        frames, graphs = graph_history['frames'], graph_history['graphs']
        last_G = graphs[-1]['G']

        v_id = frames[0][0]
        vid = '%s/%s'%(v_id.split('_')[0], v_id)

        int_dist_dict = {}
        for node in last_G.nodes():
            visits = last_G.node[node]['members']
            visits = [{'start': frames[visit['start']], 'stop': frames[visit['stop']]} for visit in visits]
            visits = [visit for visit in visits if has_feature(self.fb[vid], visit['start'][1], visit['stop'][1])]
            if len(visits)>=5:
                verb_dist, noun_dist, int_dist = self.visits_to_intdist(visits)
            else:
                verb_dist, noun_dist, int_dist = torch.zeros(125), torch.zeros(352), torch.zeros(250)

            int_dist_dict[node] = {'verb_dist':verb_dist, 'noun_dist':noun_dist, 'int_dist':int_dist}

        return int_dist_dict


    def parse_graph_data(self, graph_root, videos, out_fl):

        # frame_to_record = {}
        # int_counts = []
        # for record in self.video_infos:
        #     record.v_id = record.path.split('/')[1]
        #     for f_id in range(record.clip_start_frame, record.clip_end_frame+1):
        #         frame_to_record[(record.v_id, f_id)] = record
        #     record.uid = '%s_%s_%s'%(record.v_id, record.clip_start_frame, record.clip_end_frame)
        #     int_counts.append((record.label[0], record.label[1]))
        # self.frame_to_record = frame_to_record

        # int_counts = collections.Counter(int_counts).items()
        # int_counts = sorted(int_counts, key=lambda x: -x[1])[0:250]
        # self.int_to_idx = {interact:idx for idx, (interact, count) in enumerate(int_counts)}

        graphs = {}
        int_dist = {}
        for vid in tqdm.tqdm(videos):
            graphs[vid] = torch.load("{}/{}_graph.pth".format(graph_root, vid[4:]))
            # int_dist[vid] = self.get_graph_intdist(graphs[vid])
        print("Finished pre-loading graphs")

        for record in tqdm.tqdm(self.video_infos, total=len(self.video_infos)):
            record.cur_graph = epic_utils.get_graph(graphs[record.path], record.end_frame)
            record.next_graph = epic_utils.get_graph(graphs[record.path], record.clip_end_frame)

        torch.save({'records':self.video_infos, 'int_dist':int_dist}, out_fl)

    def get_visit_feature(self, trunk_features, start, stop, dim, v_id):
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

        #-------------------------------------------------------------------------#
        # feat = torch.load('/vision/vision_users/tushar/work/topo-ego/data/epic_frame_features_r152/%s/%s.pth'%(v_id, f_id))

        # root = '/vision/vision_users/tushar/datasets/EPIC_KITCHENS_2018/frames_rgb_flow/rgb/train/'
        # file = root+'/%s/%s/frame_%010d.jpg'%(v_id.split('_')[0], v_id, f_id)
        # img = Image.open(file).convert('RGB')
        # feat = self.transform(img)

        feat = trunk_features[f_id]

        #-------------------------------------------------------------------------#

        feat = to_tensor(feat)

        return feat

    def get_node_feats(self, graph, trunk_features, v_id):
        node_feats = []
        node_length = []
        for node in sorted(graph.nodes()):
            visits = graph.node[node]['members']
            visits = visits[-self.node_num_member:]

            feats = []
            for visit in visits:
                v_feat = self.get_visit_feature(trunk_features, visit['start'][1], visit['stop'][1], self.fb_dim, v_id)
                if v_feat is not None:
                    feats.append(v_feat)

            # Pad to 16 features
            dim = (self.fb_dim)
            # dim = (3, 224, 224)
            K = self.node_num_member
            N = min(len(feats), K)
            if len(feats)==0:
                feats = [torch.zeros(dim)]*K
                N = 1
            elif len(feats)>K:
                feats = feats[-K:]
            elif len(feats)<K:
                feats = feats + [torch.zeros(dim)]*(K-len(feats))

            feats = torch.stack(feats, 0) # (K, 2048)
            node_feats.append(feats)
            node_length.append(N)

        node_feats = torch.stack(node_feats, 0)  # (N, K, 2048)
        node_length = torch.LongTensor(node_length)

        return node_feats, node_length
    
    def process_graph_feats(self, graph, trunk_features, future_dist=None, next_graph=None, v_id=None):

        node_feats, node_length = self.get_node_feats(graph, trunk_features, v_id)

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

        g.ndata['feats'] = node_feats
        g.ndata['length'] = node_length

        cur_status = torch.zeros(len(nodes))
        cur_node = epic_utils.find_last_visit_node(graph)
        cur_status[node_to_idx[cur_node]] = 1
        g.ndata['cur_status'] = cur_status

        # nbhs
        nbhs = nx.ego_graph(graph, cur_node, radius=2, center=False).nodes()
        for nbh in nbhs:
            cur_status[node_to_idx[nbh]] = 2
        g.ndata['cur_status'] = cur_status

        next_status = torch.zeros(len(nodes))
        next_node = epic_utils.find_last_visit_node(next_graph)
        if next_node in node_to_idx and next_node!=cur_node:
            next_status[node_to_idx[next_node]] = 1
        g.ndata["next_status"] = next_status

        # if next_node == cur_node:
        #     next1_status = next_status
        # else:
        #     next1_status = 1 - cur_status
        # g.ndata["next1_status"] = next1_status


        # if future_dist is not None:
        #     g.ndata['verb_dist'] = torch.stack([future_dist[node]['vdist'] for node in nodes], 0)
        #     g.ndata['noun_dist'] = torch.stack([future_dist[node]['ndist'] for node in nodes], 0)

        return g

    def __getitem__(self, idx):
        record = self.video_infos[idx]
        data = super(EpicFeatureDatasetGFB, self).__getitem__(idx)

        g = self.process_graph_feats(record.cur_graph, self.fb[record.path], next_graph=record.next_graph, v_id=record.path.split('/')[1])

        # print ('ACTION: %s %s'%(self.verbs[record.label[0]], self.nouns[record.label[1]]))
        # for node in record.future_dist:
        #     nouns = [self.nouns[idx] for idx in record.future_dist[node]['ndist'].nonzero().squeeze(1).numpy()]
        #     verbs = [self.verbs[idx] for idx in record.future_dist[node]['vdist'].nonzero().squeeze(1).numpy()]
        #     print (node)
        #     print (nouns)
        #     print (verbs)
        #     input()


        data.update({'gfb':DC(g, stack=False, cpu_only=True)})

        return data

class EpicFeatureDatasetGFBNode(EpicFeatureDatasetGFB):
    '''
    Use precomputed generated graphs in history for each video
    '''
    def __init__(self, node_fb, **kwargs):
        super(EpicFeatureDatasetGFBNode, self).__init__(**kwargs)
        self.node_fb = mmcv.load(node_fb)

    def __getitem__(self, idx):

        record = self.video_infos[idx]
        data = super(EpicFeatureDatasetGFB, self).__getitem__(idx)

        g = self.process_graph_feats(record.cur_graph, self.node_fb[(record.path, record.end_frame)], next_graph=record.next_graph)

        data.update({'gfb':DC(g, stack=False, cpu_only=True)})

        return data
    
    def get_node_feats(self, graph, trunk_features):
        node_feats = []
        for node in sorted(graph.nodes()):
            feats = trunk_features[node]

            node_feats.append(to_tensor(feats))

        node_feats = torch.stack(node_feats, 0)  # (N, 2048)

        return node_feats


if __name__ == '__main__':

    # from mmcv.parallel import collate


    split="S1"
    # graph_root = 'data/graphs/{}/'.format(split)
    graph_root = 'data/graphs_by_entries/{}/'.format(split)
    mode=''
    phase='train'
    dataset = EpicFeatureDatasetGFB(
        ann_file='data/epic/split/{}_{}{}.csv'.format(phase, split, mode),
        feature_file="data/features/{}/".format(split) + "{}_lfb_s30.pkl".format(phase),
        test_mode=True,
        anticipation_task=True,
        fb = "data/features/{}/".format(split) + "{}_lfb_s30.pkl".format(phase),
        graph_root = graph_root
    )

    for entry in dataset:
        pass
        # input('>>')

    # imgs_per_gpu=32

    # loader = torch.utils.data.DataLoader(
    #     dataset, 
    #     batch_size=32,
    #     shuffle=False,
    #     num_workers=2,
    #     pin_memory=True,
    #     collate_fn=partial(collate, samples_per_gpu=imgs_per_gpu)
    # )
    # for batch in loader:
    #     print(batch["feature"].data[0].shape)
    #     print(batch["gt_label"].data[0].shape)

    #     break
