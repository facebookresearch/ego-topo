import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tdata
import numpy as np
import os
import collections
import re
from PIL import Image, ImageOps
import glob
import ast 
import tqdm
import h5py
import re
import torchvision.models as tmodels
import copy
import json

from ..utils import util

#-------------------------------------------------------------------------------------------------------------------#

def parse_annotations(data_dir):

    with open(f'{data_dir}/annotations/verb_idx.txt') as f:
        verbs = f.read().strip().split('\n')
        verbs = ['-'.join(line.split(' ')[:-1]).lower() for line in verbs]
    with open(f'{data_dir}/annotations/noun_idx.txt') as f:
        nouns = f.read().strip().split('\n')
        nouns = [line.split(' ')[0].lower() for line in nouns]
    with open(f'{data_dir}/annotations/action_idx.txt') as f:
        lines = f.read().strip().split('\n')
        actions = []
        for line in lines:
            split = line.split(' ')
            noun = split[-2].split(',')[0].lower()
            verb = '-'.join(split[:-2]).lower()
            actions.append((verbs.index(verb), nouns.index(noun)))

    annotations = {'verbs':verbs, 'nouns':nouns, 'actions':actions}

    def parse_interaction_annotatons(lines):
        entries = []
        for line in lines:

            split = line.split(' ')
            clip, action, verb, noun = split[:4]
            noun2 = split[4] if len(split)==5 else '0'

            P, R, dish, clip_start, clip_stop, fstart, fstop = clip.split('-')
            v_id = f'{P}-{R}-{dish}'
            entry = {'v_id':v_id,
                     'start':int(fstart[1:]), 'stop':int(fstop[1:]),
                     'clip_start':int(clip_start), 'clip_stop':int(clip_stop),
                     'verb':int(verb)-1, 'noun':int(noun)-1, 'noun2':int(noun2)-1, 'action':int(action)-1}
            entries.append(entry)

        entries = sorted(entries, key=lambda entry:entry['start'])

        return entries

    lines = open(f'{data_dir}/annotations/train_split1.txt').read().strip().split('\n')
    lines += open(f'{data_dir}/annotations/test_split1.txt').read().strip().split('\n')
    interactions = parse_interaction_annotatons(lines)
    annotations['interactions'] = interactions


    videos = set([entry['v_id'] for entry in interactions])
    annotations['videos'] = sorted(videos)

    # interactions in train/test come from the same videos. Use an alternate split
    train_test_splits = json.load(open(f'{data_dir}/train_test_splits.json'))
    annotations.update(train_test_splits)

    video_lengths = collections.defaultdict(int)
    for entry in interactions:
        video_lengths[entry['v_id']] = max(video_lengths[entry['v_id']], entry['stop'])
    annotations['vid_lengths'] = video_lengths

    torch.save(annotations, 'build_graph/data/gtea/gtea_data.pth')

#-------------------------------------------------------------------------------------------------------------------#


class GTEA(torch.utils.data.Dataset):

    def __init__(self, root):
        super().__init__()
        self.root = root
        self.fps = 24

        if not os.path.exists('build_graph/data/gtea/gtea_data.pth'):
            parse_annotations(self.root)
            print ('Annotations created!')

        self.annotations =  torch.load('build_graph/data/gtea/gtea_data.pth')
        self.interactions = self.annotations['interactions']
        self.verbs, self.nouns = self.annotations['verbs'], self.annotations['nouns']
        self.train_vids, self.val_vids = self.annotations['train_vids'], self.annotations['val_vids']

    def frame_path(self, img):
        v_id, f_id = img
        file = f'{self.root}/frames/{v_id}/frame_{f_id:010d}.jpg'
        return file        

    def load_image(self, img):
        file = self.frame_path(img)
        img = Image.open(file).convert('RGB')
        return img


class GTEAInteractions(GTEA):

    def __init__(self, root, split, clip_len):
        super().__init__(root)

        self.split = split
        self.clip_len = clip_len

        self.train_data = self.parse_data_for_split(self.train_vids)
        self.val_data = self.parse_data_for_split(self.val_vids)
        self.data = self.train_data if self.split=='train' else self.val_data
        print (f'Train data: {len(self.train_data)} | Val data: {len(self.val_data)}')
        
        # self.clip_transform = util.clip_transform(self.split, self.clip_len)    

    def parse_data_for_split(self, videos):
        videos = set(videos)
        clips = []
        for entry in self.interactions:

            if entry['v_id'] not in videos:
                continue

            clip = dict(entry)
            frames = [(clip['v_id'], f_id) for f_id in range(clip['start'], clip['stop']+1)] # frames @ 24fps
            uid = '{}_{}_{}_{}_{}'.format(clip['v_id'], clip['start'], clip['stop'], clip['verb'], clip['noun'])
            clip.update({'frames':frames, 'uid': uid})
            clips.append(clip)
        return clips

    def sample(self, imgs):
        
        if len(imgs)>self.clip_len:
            if self.split=='train': # random sample
                offset = np.random.randint(0, len(imgs)-self.clip_len)
                indices = slice(offset, offset+self.clip_len)
            elif self.split=='val': # center crop
                offset = len(imgs)//2 - self.clip_len//2
                indices = slice(offset, offset+self.clip_len)
        else:
            indices = slice(0, len(imgs))

        return indices

    def __getitem__(self, index):

        entry = self.data[index]
        frame_sample = self.sample(entry['frames'])
        frames = [self.load_image(f) for f in entry['frames'][frame_sample]]
        frames = self.clip_transform(frames) # (T, 3, 224, 224)
        frames = frames.permute(1, 0, 2, 3)
        instance = {'frames':frames, 'verb':entry['verb'], 'noun':entry['noun']}

        return instance

    def __len__(self):
        return len(self.data)


#------------------------------------------------------------------------------------------------------------------------------#

class GTEAFrames(GTEA):

    def __init__(self, root):
        super().__init__(root)

        vid_lengths = self.annotations['vid_lengths']
        frames = []
        for v_id in vid_lengths:
            frames += [(v_id, f_id) for f_id in range(1, vid_lengths[v_id]+1)]
        self.frames = frames
        self.transform = util.default_transform('val')

        self.keys = ['%s/%d'%(v_id, f_id) for v_id, f_id in self.frames]
        self.keys = np.array(self.keys, dtype='S')

    def __getitem__(self, index):
        img = self.load_image(self.frames[index])
        img = self.transform(img)
        return {'frame': img}

    def __len__(self):
        return len(self.frames)

#------------------------------------------------------------------------------------------------------------------------------#



