import numpy as np
import torch
import glob
import collections
import tqdm
import os
import math
import h5py 
import argparse
import shutil
import torch.nn as nn

from ..data import epic, gtea
from ..localization_network.model import SiameseR18_5MLP
from ..utils import util

# subsampled frames @6fps for graph construction
class FrameDsetSubsampled:

    def __init__(self, dset):

        if dset=='gtea':
            self.dset = gtea.GTEAInteractions('build_graph/data/gtea', 'val', 32)
        elif dset=='epic':
            self.dset = epic.EPICInteractions('build_graph/data/epic', 'val', 32)

        # subsample frames to 6fps for graph generation
        subsample =  self.dset.fps//6
        frames = []
        for v_id in self.dset.train_vids + self.dset.val_vids:
            vid_length = self.dset.annotations['vid_lengths'][v_id]
            frames += [(v_id, f_id) for f_id in range(1, vid_length+1, subsample)]
        self.frames = frames
        self.transform = util.default_transform('val')

        print (f'Generating localization net features for {len(self.frames)} frames')

    def __getitem__(self, index):
        img = self.dset.load_image(self.frames[index])
        img = self.transform(img)
        return {'frame': img}

    def __len__(self):
        return len(self.frames)

def generate_features(args):

    net = SiameseR18_5MLP()
    checkpoint = torch.load(args.load, map_location='cpu')
    net.load_state_dict(checkpoint['net'])

    # save the classifier head separately
    clf_weights = {k:v for k,v in checkpoint['net'].items() if 'compare' in k}
    torch.save({'net':clf_weights}, f'{os.path.dirname(args.load)}/head.pth')

    trunk = net.trunk # only keep the resnet50 trunk
    if args.parallel:
    	trunk = nn.DataParallel(trunk)
    trunk.eval().cuda()

    dataset = FrameDsetSubsampled(args.dset)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False, num_workers=8)

    feats = []
    for batch in tqdm.tqdm(loader, total=len(loader)):
        with torch.no_grad():
            out = trunk(batch['frame'].cuda())
        out = out.cpu()
        feats.append(out)
    feats = torch.cat(feats, 0)
    cache = dict(zip(dataset.frames, feats))

    torch.save(cache, f'build_graph/data/{args.dset}/locnet_trunk_feats.pth')


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--load', default=None)
    parser.add_argument('--dset', default=None)
    parser.add_argument('--parallel', action ='store_true', default=False)
    args = parser.parse_args()

    generate_features(args)
