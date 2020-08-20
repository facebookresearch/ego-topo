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

from ..utils import util

#-------------------------------------------------------------------------------------------------------------------#

def parse_annotations(data_dir):

    # get the list of object and action classes
    with open(f'{data_dir}/annotations/EPIC_verb_classes.csv') as f:
        verbs = f.read().strip().split('\n')[1:]
        verbs = [line.split(',')[1] for line in verbs]
    with open(f'{data_dir}/annotations/EPIC_noun_classes.csv') as f:
        nouns = f.read().strip().split('\n')[1:]
        nouns = [line.split(',')[1] for line in nouns]

    annotations = {'nouns':nouns, 'verbs':verbs}

    # parse the interactions
    interactions = []
    interaction_labels = open(f'{data_dir}/annotations/EPIC_train_action_labels.csv').read().strip().split('\n')[1:]
    for line in interaction_labels:
        split = line.split(',')
        v_id = split[2]
        start_time, stop_time, start, stop = split[4:8]
        verb, verb_class, noun, noun_class = split[8:12]
        interactions.append({'v_id':v_id, 'start_time':start_time, 'stop_time':stop_time, 'start':int(start), 'stop':int(stop), 'verb':int(verb_class), 'noun':int(noun_class)})
    annotations['interactions'] = interactions

    videos = set([entry['v_id'] for entry in interactions])
    annotations['videos'] = sorted(videos)

    # S1: Seen Kitchens split - 80:20 split for train/val
    vid_by_person = collections.defaultdict(list)
    for v_id in videos:
        vid_by_person[v_id.split('_')[0]].append(v_id)

    train_vids, val_vids = [], []
    for person in vid_by_person:
        vids = sorted(vid_by_person[person])
        offset = int(0.8*len(vids))
        train_vids += vids[:offset]
        val_vids += vids[offset:]
        
    annotations.update({'train_vids':train_vids, 'val_vids':val_vids})


    video_lengths = collections.defaultdict(int)
    for entry in interactions:
        video_lengths[entry['v_id']] = max(video_lengths[entry['v_id']], entry['stop'])
    annotations['vid_lengths'] = video_lengths

    torch.save(annotations, 'build_graph/data/epic/epic_data.pth')

#-------------------------------------------------------------------------------------------------------------------#

class EPIC(torch.utils.data.Dataset):

    def __init__(self, root):
        super().__init__()
        self.root = root
        self.fps = 60
        
        if not os.path.exists('build_graph/data/epic/epic_data.pth'):
            parse_annotations(self.root)
            print ('Annotations created!')

        self.annotations =  torch.load('build_graph/data/epic/epic_data.pth')
        self.interactions = self.annotations['interactions']
        self.verbs, self.nouns = self.annotations['verbs'], self.annotations['nouns']
        self.train_vids, self.val_vids = self.annotations['train_vids'], self.annotations['val_vids']

    def frame_path(self, img):
        v_id, f_id = img
        p_id = v_id.split('_')[0]
        # file = f'{self.root}/frames/train/{p_id}/{v_id}/frame_{f_id:010d}.jpg' # orig
        file = f'{self.root}/frames/train/{v_id}/frame_{f_id:010d}.jpg' # devfair
        return file        

    def load_image(self, img):
        file = self.frame_path(img)
        img = Image.open(file).convert('RGB')
        return img


#------------------------------------------------------------------------------------------------------------------------------#

class EPICInteractions(EPIC):

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
            frames = [(clip['v_id'], f_id) for f_id in range(clip['start'], clip['stop']+1, 2)] # frames @ 30fps
            uid = '{}_{}_{}_{}_{}'.format(clip['v_id'], clip['start_time'], clip['stop_time'], clip['verb'], clip['noun'])
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

class EPICFrames(EPIC):

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

