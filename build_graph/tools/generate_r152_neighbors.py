import numpy as np
import collections
import torch
from data import epic
import os
import glob
from utils import util
from PIL import Image, ImageOps
import h5py
import tqdm
import itertools
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import euclidean_distances
import re
from torchvision import models as tmodels
import torch
import torch.nn as nn

import data
from data import epic, gtea

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dset', default=None)
args = parser.parse_args()

class ClipFeats:

    def __init__(self, dset):
        self.data = dset.train_data + dset.val_data
        self.load_image = dset.load_image
        self.transform = util.default_transform('val')

    def __getitem__(self, index):
        entry = self.data[index]
        frames = [self.load_image(f) for f in entry['frames']]
        frames = [self.transform(f) for f in frames]
        frames = torch.stack(frames, 0)
        return {'frames': frames, 'uid':entry['uid']}

    def __len__(self):
        return len(self.data)

def featurize(frames, net, bs=512):
    feats = []
    for idx in range(0, len(frames), bs):
        batch = frames[idx:idx+bs]
        with torch.no_grad():
            feat = net(batch.cuda()).cpu()
        feats.append(feat)
    feats = torch.cat(feats, 0)
    return feats.mean(0)


# for each interaction clip, calculate average r152 features
# returns: {uid: 2048D feature}
# On 4 K40s: GTEA (~1 hr) | EPIC (~3 hr)
def generate_avg_interaction_features():

    net = tmodels.resnet152(pretrained=True)
    net.fc = nn.Identity()
    net = nn.DataParallel(net)
    net.cuda().eval()

    if args.dset=='epic':
        dset = epic.EPICInteractions('data/epic', 'val', 32)
    elif args.dset=='gtea':
        dset = gtea.GTEAInteractions('data/gtea', 'val', 32)
    dset = ClipFeats(dset)
    loader = torch.utils.data.DataLoader(dset, batch_size=1, shuffle=False, num_workers=8)

    avg_feats = {}
    for batch in tqdm.tqdm(loader, total=len(loader)):
        feats = featurize(batch['frames'][0], net)
        avg_feats[batch['uid'][0]] = feats

    torch.save(avg_feats, f'build_graph/data/{args.dset}/avg_feats_r152.pth')

# generate pairwise distance matrix between every clip in the dataset
# distance = feature similarity of average resnet152 features computed in generate_avg_interaction_features()
# restrict to only clips from the same kitchen (EPIC) to save time (< 2 mins to run)
def generate_pairwise_clip_distance():
    if args.dset=='epic':
        generate_pairwise_clip_distance_epic()
    elif args.dset=='gtea':
        generate_pairwise_clip_distance_gtea()


def generate_pairwise_clip_distance_epic():

    def generate_pdist_matrices(videos, split):

        pattern = re.compile('|'.join(videos))
        uids = [uid for uid in all_uids if pattern.match(uid)]

        avg_feats = torch.stack([all_avg_feats[uid] for uid in uids], 0).numpy()
        pdist_dense = euclidean_distances(avg_feats)
        print (split, pdist_dense.shape)

        # Matches within same kitchen
        pdist = np.array(pdist_dense)
        uid_to_idx = {uid:idx for idx, uid in enumerate(uids)}

        z_idx = {}
        for idxA in range(len(uids)):
            kitchen = uids[idxA][:3]
            if kitchen not in z_idx:
                z_idx[kitchen] = [idxB for idxB in range(len(uids)) if uids[idxB][:3]!=kitchen]

        for idx in tqdm.tqdm(range(len(uids))):
            kitchen = uids[idx][:3]
            pdist[idx][z_idx[kitchen]] = 0
        pdist = csr_matrix(pdist)

        out_fl = f'build_graph/data/{args.dset}/{split}_pdist_r152_samek.pth'
        torch.save({'uids':uids, 'uid_to_idx':uid_to_idx, 'pdist':pdist}, out_fl)

    all_avg_feats = torch.load(f'build_graph/data/{args.dset}/avg_feats_r152.pth')
    all_uids = sorted(all_avg_feats.keys())
    
    dset = epic.EPICInteractions('data/epic', 'val', 32)
    generate_pdist_matrices(dset.train_vids, 'train')
    generate_pdist_matrices(dset.val_vids, 'val')

def generate_pairwise_clip_distance_gtea():

    def generate_pdist_matrices(data, split):

        uids = [entry['uid'] for entry in data]
        avg_feats = torch.stack([all_avg_feats[uid] for uid in uids], 0).numpy()
        pdist = euclidean_distances(avg_feats)
        print (split, pdist.shape)

        # Matches within same kitchen
        uid_to_idx = {uid:idx for idx, uid in enumerate(uids)}
        pdist = csr_matrix(pdist)

        out_fl = f'build_graph/data/{args.dset}/{split}_pdist_r152_samek.pth'
        torch.save({'uids':uids, 'uid_to_idx':uid_to_idx, 'pdist':pdist}, out_fl)

    all_avg_feats = torch.load(f'build_graph/data/{args.dset}/avg_feats_r152.pth')

    dset = gtea.GTEAInteractions('data/gtea', 'val', 32)
    generate_pdist_matrices(dset.train_data, 'train')
    generate_pdist_matrices(dset.val_data, 'val')


#-------------------------------------------------------------------------------------------------------------------#
if __name__=='__main__':
    generate_avg_interaction_features()
    generate_pairwise_clip_distance()


