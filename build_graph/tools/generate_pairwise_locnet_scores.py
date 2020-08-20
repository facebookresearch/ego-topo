import torch
import torch.nn as nn
import os
import itertools 
import tqdm
import numpy as np
import argparse

from ..utils import util
from ..data import gtea, epic
from ..localization_network.model import R18_5MLP


parser = argparse.ArgumentParser()
parser.add_argument('--dset', default=None)
parser.add_argument('--load', default=None)
parser.add_argument('--v_id', default=None)
parser.add_argument('--cv_dir', default=None)
parser.add_argument('--parallel', action ='store_true', default=False)
args = parser.parse_args()


class PairwiseFrames:

    def __init__(self, dset, v_id, feat_source):
        
        if dset=='epic':
            dataset = epic.EPICInteractions('build_graph/data/epic', 'val', 32)
        elif dset=='gtea':
            dataset = gtea.GTEAInteractions('build_graph/data/gtea', 'val', 32)
            
        frames = [(v_id, f_id) for f_id in range(1, dataset.annotations['vid_lengths'][v_id] + 1, dataset.fps//6)]
        self.data = list(itertools.combinations(frames, 2))
        self.feat_cache = torch.load(feat_source)
        print (f'Generating pairwise scores for {len(frames)} frames --> {len(self.data)} pairs')


    def __getitem__(self, index):
        imgA, imgB = self.data[index]
        imgA = self.feat_cache[imgA]
        imgB = self.feat_cache[imgB]
        return {'imgA':imgA, 'imgB':imgB}

    def __len__(self):
        return len(self.data)

def compute_pairwise_scores():

    net = R18_5MLP()
    checkpoint = torch.load(args.load, map_location='cpu')
    net.load_state_dict(checkpoint['net'])
    if args.parallel:
        net = nn.DataParallel(net)
    net.cuda().eval()
    print (f'Loaded checkpoint from {os.path.basename(args.load)}')

    feat_source = f'build_graph/data/{args.dset}/locnet_trunk_feats.pth'
    dataset = PairwiseFrames(args.dset, args.v_id, feat_source) 
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False, num_workers=8)

    scores = []
    for batch in tqdm.tqdm(loader, total=len(loader)):
        batch = util.batch_cuda(batch)
        with torch.no_grad():
            pred, _ = net.forward(batch, softmax=True)
        scores.append(pred.cpu())
    scores = torch.cat(scores, 0).numpy()

    pairwise_score_fl = f'build_graph/data/{args.dset}/locnet_pairwise_scores/{args.v_id}_scores.pth'
    os.makedirs(os.path.dirname(pairwise_score_fl), exist_ok=True)
    mm = np.memmap(pairwise_score_fl, dtype='float32', mode='w+', shape=scores.shape)
    mm[:] = scores[:]
    del mm


if __name__=='__main__':
    compute_pairwise_scores()
    