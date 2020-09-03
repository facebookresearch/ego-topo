# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import numpy as np
import collections
import torch
import os
import glob
from PIL import Image, ImageOps
import h5py
import tqdm
import itertools

from ..data import epic, gtea
from ..utils import util

#-------------------------------------------------------------------------------------------#

# Base class that implements all selection strategies
class RetrievalBase:

    def __init__(self):
        self.min_inliers = 10 # number of inliers for homography matching above which frames are considered "similar" 
        self.T_threshold = 20 # number of clips after which a clip is considered "far in time"
        self.r152_threshold = 20 # feature dot product above which frames are considered "dissimilar"

    # select a frame from the same clip
    def same_interaction(self, indexA):
        entryB = self.data[indexA]
        frameB = self.rs.randint(0, len(entryB['frames']))
        frameB = entryB['frames'][frameB]
        return frameB, 'T' 

    # select a clip index T timesteps away from clipA (on either side)
    def get_far_idx(self, indexA, T, ub=None):
        ub = ub or len(self.data)

        if indexA<=T:
            side = 'right'
        elif indexA>=ub - T:
            side = 'left'
        else:
            side = 'left' if self.rs.rand()<0.5 else 'right'

        if side=='right':
            indexB = self.rs.randint(indexA + T, ub)
        elif side=='left':
            indexB = self.rs.randint(0, indexA - T)

        return indexB

    # select an unannotated frame (background frame)
    def bg_frame(self, indexA):
        entryA = self.data[indexA]
        frameA = entryA['frames'][len(entryA['frames'])//2]
        frameB = frameA
        while np.abs(frameB[1] - frameA[1])<5*self.FPS:
            frameB = self.bg_frames[entryA['v_id']][np.random.randint(len(self.bg_frames))]
        return frameB, 'BG' 


    # select frame from a different interaction, at least T interactions away.
    def far_interaction(self, indexA, T):
        indexB = self.get_far_idx(indexA, T)
        entryB = self.data[indexB]
        frameB = self.rs.randint(0, len(entryB['frames']))
        frameB = entryB['frames'][frameB]
        return frameB, 'T'

    # select a clip that is far away AND visually dissimular
    def far_interaction_visually_dissimilar(self, indexA, T, pdist, threshold):

        uidxA = pdist['uid_to_idx'][self.data[indexA]['uid']]
        while True:
            indexB = self.get_far_idx(indexA, T)
            uidxB = pdist['uid_to_idx'][self.data[indexB]['uid']]
            if pdist['pdist'][uidxA, uidxB] == 0 or pdist['pdist'][uidxA, uidxB]>threshold:
                break
        entryB = self.data[indexB]
        frameB = self.rs.randint(0, len(entryB['frames']))
        frameB = entryB['frames'][frameB]
        return frameB, f'T: {pdist["pdist"][uidxA, uidxB]:3f}'

    # select a clip that is far away, visually dissimilar, but from the same kitchen (in EPIC)
    def far_interaction_same_kitchen_visually_dissimilar(self, indexA, T, pdist, threshold):
        entryA = self.data[indexA]
        candidates = [entry for entry in self.data if entryA['v_id'].split('_')[0] in entry['v_id']]
        indexA_new = [idx for idx in range(len(candidates)) if candidates[idx]['uid']==entryA['uid']][0] 
        if len(candidates)<2*T+2:
            # print ('Not enough candidates: OOPS')
            return self.far_interaction(indexA, T)

        uidxA = pdist['uid_to_idx'][self.data[indexA]['uid']]
        timeout = 100
        for t in range(timeout):
            indexB = self.get_far_idx(indexA_new, T, len(candidates))
            uidxB = pdist['uid_to_idx'][self.data[indexB]['uid']]
            if pdist['pdist'][uidxA, uidxB] == 0 or pdist['pdist'][uidxA, uidxB]>threshold:
                break
        if t==timeout-1:
            # print ('Timeout: OOPS')
            return self.far_interaction(indexA, T)

        entryB = candidates[indexB]
        frameB = self.rs.randint(0, len(entryB['frames']))
        frameB = entryB['frames'][frameB]
        return frameB, f'T: {pdist["pdist"][uidxA, uidxB]:3f}'

    # select an entry that has >threshold inliers after estimating homography
    def sim_homography(self, indexA, pdist, threshold):
        entryA = self.data[indexA]
        uidx = pdist['uid_to_idx'][entryA['uid']]
        ninds = pdist['pdist'][uidx].nonzero()[1]
        candidates = [{'uid':pdist['uids'][nidx], 'score':pdist['pdist'][uidx, nidx]} for nidx in ninds]
        candidates = list(filter(lambda nbh: nbh['score']>threshold, candidates))
        
        timeout = 100
        for t in range(timeout):
            candidate = candidates[self.rs.randint(len(candidates))]
            entryB = self.uid_to_entry[candidate['uid']]
            if np.abs(entryB['frames'][len(entryB['frames'])//2][1]-entryA['frames'][len(entryA['frames'])//2][1]) > 100:
                break
        if t==timeout-1:
            # print ('Timeout: OOPS')
            return self.same_interaction(indexA)

        entryB = self.uid_to_entry[candidate['uid']]
        frameB = self.rs.randint(0, len(entryB['frames']))
        frameB = entryB['frames'][frameB]
        return frameB, f'SP: {candidate["score"]:3f}'


    # subclasses should decide how to actually pick among selection strategies
    # for positive and negative samples
    def select_positive(self):
        raise NotImplementedError

    def select_negative(self):
        raise NotImplementedError


    # load pairwise distance files and keep instances that satisfy constraints (keep_fn)
    def parse_pdist(self, pdist_fl, keep_fn):
        pdist = torch.load(pdist_fl)
        pdist['pdist'] = pdist['pdist'].todense()

        # find uids that actually have neighbors according to our thresholds
        # don't waste time returning None for the others
        keep = keep_fn(pdist['pdist']).sum(1)
        keep = keep.nonzero()[0]
        pdist['domain'] = set([pdist['uids'][idx] for idx in keep])

        print (f'Keeping {len(pdist["domain"])}/{len(self.data)} uids for {os.path.basename(pdist_fl)}')
        
        return pdist

    def load_transform(self, frame):
        frame = self.load_image(frame)
        frame = self.transform(frame)
        return frame

    def __getitem__(self, index):

        entryA = self.data[index]

        idxA = self.rs.randint(0, len(entryA['frames']))

        frameA = entryA['frames'][idxA]
        label = 1 if self.rs.rand()<0.5 else 0
        frameB, src = self.select_positive(index) if label==1 else self.select_negative(index)

        # arbitrary order for frameA, frameB
        if self.rs.rand()<0.5:
            frameA, frameB = frameB, frameA

        meta = (frameA, frameB, label, src)
        imgA = self.load_transform(frameA)
        imgB = self.load_transform(frameB)

        instance = {'imgA':imgA, 'imgB':imgB, 'label':label, 'meta':meta}
        return instance

    def __len__(self):
        return len(self.data)


#-------------------------------------------------------------------------------------------#

class EPICPairs(RetrievalBase):
    def __init__(self, root, split):
        super().__init__()

        dset = epic.EPICInteractions(root, split, 32)
        for key in ['split', 'annotations', 'data', 'train_vids', 'val_vids', 'train_data', 'val_data', 'load_image']:
            setattr(self, key, getattr(dset, key))

        self.retr_dir = 'build_graph/data/epic/'

        seed = 8275 if self.split=='val' else None
        self.rs = np.random.RandomState(seed)
        self.transform = util.default_transform('val')
        self.uid_to_entry = {entry['uid']:entry for entry in self.data}

        videos = self.train_vids if self.split=='train' else self.val_vids
        bg_frames = {v_id:set([(v_id, f_id) for f_id in range(1, self.annotations['vid_lengths'][v_id])]) for v_id in videos}
        for entry in self.data:
            bg_frames[entry['v_id']] -= set([(entry['v_id'], f_id) for f_id in range(entry['start'], entry['stop']+1)])
        self.bg_frames = bg_frames

        self.FPS = 60

        self.pdist = {}
        self.pdist['homography'] = self.parse_pdist(pdist_fl = f'{self.retr_dir}/{self.split}_homography_inliers.pth',
                                                    keep_fn = lambda mat: (mat>self.min_inliers))
        self.pdist['r152'] = self.parse_pdist(pdist_fl = f'{self.retr_dir}/{self.split}_pdist_r152_samek.pth',
                                              keep_fn = lambda mat: (mat==mat))

        self.data = [entry for entry in self.data if entry['uid'] in self.pdist['homography']['domain']]
        print (f'Keeping {len(self.data)} entries in {self.split}')

    def select_positive(self, index):
        p = [0.5, 0.5]
        opt = self.rs.choice([0, 1], p=p)
        if opt==0:
            frameB, src = self.same_interaction(index)
        elif opt==1:
            frameB, src = self.sim_homography(index, self.pdist['homography'], self.min_inliers)
        return frameB, src

    def select_negative(self, index):
        frameB, src = self.far_interaction_same_kitchen_visually_dissimilar(index, self.T_threshold, self.pdist['r152'], self.r152_threshold)
        return frameB, src



class GTEAPairs(RetrievalBase):
    def __init__(self, root, split):
        super().__init__()

        dset = gtea.GTEAInteractions(root, split, 32)
        for key in ['split', 'annotations', 'data', 'train_vids', 'val_vids', 'train_data', 'val_data', 'load_image']:
            setattr(self, key, getattr(dset, key))

        self.retr_dir = 'build_graph/data/gtea/'

        seed = 8275 if self.split=='val' else None
        self.rs = np.random.RandomState(seed)
        self.transform = util.default_transform('val')
        self.uid_to_entry = {entry['uid']:entry for entry in self.train_data + self.val_data}

        videos = self.train_vids if self.split=='train' else self.val_vids
        bg_frames = {v_id:set([(v_id, f_id) for f_id in range(1, self.annotations['vid_lengths'][v_id])]) for v_id in videos}
        for entry in self.data:
            bg_frames[entry['v_id']] -= set([(entry['v_id'], f_id) for f_id in range(entry['start'], entry['stop']+1)])
        self.bg_frames = {k:list(v) for k,v in bg_frames.items()}

        self.FPS = 24
        self.pdist = {}
        self.pdist['homography'] = self.parse_pdist(pdist_fl = f'{self.retr_dir}/{self.split}_homography_inliers.pth',
                                                    keep_fn = lambda mat: (mat>self.min_inliers))
        self.pdist['r152'] = self.parse_pdist(pdist_fl = f'{self.retr_dir}/{self.split}_pdist_r152_samek.pth',
                                              keep_fn = lambda mat: (mat==mat))
        self.data = [entry for entry in self.data if entry['uid'] in self.pdist['homography']['domain']]
        print (f'Keeping {len(self.data)} entries in {self.split}')

    def select_positive(self, index):
        p = [0.5, 0.5]
        opt = self.rs.choice([0, 1], p=p)
        if opt==0:
            frameB, src = self.same_interaction(index)
        elif opt==1:
            frameB, src = self.sim_homography(index, self.pdist['homography'], self.min_inliers)
        return frameB, src

    def select_negative(self, index):
        p = [0.5, 0.5]
        opt = self.rs.choice([0, 1], p=p)
        if opt==0:
            frameB, src = self.far_interaction_visually_dissimilar(index, self.T_threshold, self.pdist['r152'], self.r152_threshold)
        elif opt==1:
            frameB, src = self.bg_frame(index)
        return frameB, src


if __name__=='__main__':
    from torchvision.utils import make_grid

    dataset = EPICPairs('data/epic', 'train')
    # dataset = GTEAPairs('data/gtea', 'train')

    viz = []
    for idx, entry in enumerate(dataset.data):

        instance = dataset[idx]
        imgA = util.unnormalize(instance['imgA'], 'imagenet')
        imgB = util.unnormalize(instance['imgB'], 'imagenet')

        color = 'green' if instance['label']==1 else 'red'
        imgA, imgB = util.add_border(imgA, color), util.add_border(imgB, color)
        viz += [imgA, imgB]

        print (instance['meta'])

        if len(viz)==20:
            viz = viz[0::2] + viz[1::2]
            grid = make_grid(viz, nrow=10)
            util.show_wait(grid)
            viz = []
            print ('-'*10)
