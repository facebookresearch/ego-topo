import bisect
import copy
from collections import Sequence

import numpy as np
import torch

import mmcv
from mmcv.runner import obj_from_dict

from .. import datasets


class EpicRawFramesRecord(object):
    def __init__(self, row, clip_range=None):
        self._data = row
        self.clip_range = clip_range

    @property
    def path(self):
        return self._data[0]

    @property
    def start_frame(self):
        return int(self._data[1])

    @property
    def end_frame(self):
        return int(self._data[2])

    @property
    def label(self):
        return [int(x) for x in self._data[3:]]

    @property
    def num_frames(self):
        return self.end_frame - self.start_frame + 1

    @property
    def clip_start_frame(self):
        return int(self._data[1]) if self.clip_range is None else int(self.clip_range[0])

    @property
    def clip_end_frame(self):
        return int(self._data[2]) if self.clip_range is None else int(self.clip_range[1])


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError('type {} cannot be converted to tensor.'.format(
            type(data)))


def get_trimmed_dataset(data_cfg):
    if 'ann_file' not in data_cfg:
        ann_files = [None]
        num_dset = 1
    elif isinstance(data_cfg['ann_file'], (list, tuple)):
        ann_files = data_cfg['ann_file']
        num_dset = len(ann_files)
    else:
        ann_files = [data_cfg['ann_file']]
        num_dset = 1

    if 'img_prefix' not in data_cfg:
        img_prefixes = [None]
    elif isinstance(data_cfg['img_prefix'], (list, tuple)):
        img_prefixes = data_cfg['img_prefix']
    else:
        img_prefixes = [data_cfg['img_prefix']]
    assert len(img_prefixes) == num_dset

    dsets = []
    for i in range(num_dset):
        data_info = copy.deepcopy(data_cfg)
        if ann_files[i] is not None:
            data_info['ann_file'] = ann_files[i]
        if img_prefixes[i] is not None:
            data_info['img_prefix'] = img_prefixes[i]
        dset = obj_from_dict(data_info, datasets)
        dsets.append(dset)

    if len(dsets) > 1:
        raise ValueError("Not implemented yet")
    else:
        dset = dsets[0]

    return dset


def get_visit_feature(trunk_features, start, stop, dim):
    feats = []
    for fid in range(start - 31, stop - 31):
        if fid in trunk_features.keys():
            feats.append(to_tensor(trunk_features[fid]))

    if len(feats) == 0:
        return torch.zeros((1, dim))
    else:
        return torch.stack(feats, 0).mean(0, keepdim=True)
    

def get_visit_labels(annos, keys, start, stop):
    idx = bisect.bisect(keys, start)

    assert idx == len(annos) or keys[idx] > start

    if idx > 0 and annos[idx - 1].end_frame > start:
        return annos[idx - 1].label
    elif idx < len(annos) and annos[idx].start_frame < stop:
        return annos[idx].label
    else:
      #  print(idx, start, stop)
        # if (idx > 0):
        #    # print(annos[idx - 1].start_frame, annos[idx - 1].end_frame)
        # if idx < len(annos):
        #    print(annos[idx].start_frame, annos[idx].end_frame)
        return [-1, -1]


def load_graphs(graph_root, videos):
    graphs = {}
    for vid in videos:
        graphs[vid] = torch.load("{}/{}_graph.pth".format(graph_root, vid[4:]))

    print("Finished pre-loading graphs")
    return graphs


def get_graph(graph_history, frame_idx):
    frames, graphs = graph_history['frames'], graph_history['graphs']

    keys = [x[1] for x in frames]
    idx = bisect.bisect(keys, frame_idx) - 1

    assert idx >= 0 and keys[idx] <= frame_idx and idx == graphs[idx]['frame'], "{} {} {}".format(idx, keys[idx], frame_idx)

    graph = graphs[idx]

    G = copy.deepcopy(graph['G'])
    # G.last_state = graph['last_state']
    for node in G.nodes():
        visits = G.node[node]['members']
        visits = [{'start': frames[visit['start']], 'stop': frames[visit['stop']]} for visit in
                    visits]
        G.node[node]['members'] = visits

    return G


def find_last_visit_node(graph):
    last = None
    for node in graph.nodes():
        if last is None or graph.node[node]["members"][-1]["stop"][1] > graph.node[last]["members"][-1]["stop"][1]:
            last = node
    
    return last