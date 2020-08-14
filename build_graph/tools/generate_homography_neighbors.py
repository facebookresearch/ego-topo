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
import re

import data
from data import epic, gtea


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dset', default=None)
args = parser.parse_args()


def generate_inliers_matrices(all_uids, neighbors, videos, split):

    pattern = re.compile('|'.join(videos))
    uids = [uid for uid in all_uids if pattern.match(uid)]
    uids = sorted(uids)
    uid_to_idx = {uid:idx for idx, uid in enumerate(uids)}

    inliers_dense = np.zeros((len(uids), len(uids)))

    #-------------------------------------------------------------------------------------------------------#

    inliers = np.array(inliers_dense)
    for idx, uid in tqdm.tqdm(enumerate(uids), total=len(uids)):
        for nbh in neighbors[uid]:
            if nbh['nbh'] in uid_to_idx:
                nbh_idx = uid_to_idx[nbh['nbh']]
                inliers[idx][nbh_idx] = nbh['inliers']
    inliers = csr_matrix(inliers)
  
    torch.save({'uids':uids, 'uid_to_idx':uid_to_idx, 'pdist':inliers}, f'build_graph/data/{args.dset}/{split}_homography_inliers.pth')


def generate_nbhs(homography_file):

    if not os.path.exists(homography_file):
        print (f'{homography_file} MISSING')
        return {}

    hom_data = torch.load(homography_file)
    uids = hom_data['uids']
    homographies = hom_data['homography']

    neighbors = collections.defaultdict(list)
    for idx, (i,j) in enumerate(itertools.combinations(range(len(uids)), 2)):
        uid1, uid2 = uids[i], uids[j]
        homography = homographies[idx]

        if homography['inliers'] is None:
            continue

        matches, inliers = homography['matches'].shape[1], homography['inliers'].shape[0]
        neighbors[uid1].append({'nbh':uid2, 'matches':matches, 'inliers':inliers})
        neighbors[uid2].append({'nbh':uid1, 'matches':matches, 'inliers':inliers})

    return neighbors

def generate_pairwise_inliers():

    if args.dset=='gtea':
        dset = gtea.GTEAInteractions('data/gtea', 'val', 32)
    elif args.dset=='epic':
        dset = epic.EPICInteractions('data/epic', 'val', 32)

    neighbors = collections.defaultdict(list)
    for homography_file in glob.glob(f'build_graph/data/{args.dset}/matches/*/*.pth'):
        nbhs = generate_nbhs(homography_file)
        for key in nbhs:
            neighbors[key] += nbhs[key]

    all_uids = sorted(neighbors.keys())
    generate_inliers_matrices(all_uids, neighbors, dset.train_vids, 'train')
    generate_inliers_matrices(all_uids, neighbors, dset.val_vids, 'val')


#-------------------------------------------------------------------------------------------------------------------#
if __name__=='__main__':
    # call using .sh script to do this in parallel
    # ideally by submitting each run to cluster nodes
    # > bash generate_sp_matches.sh epic val
    generate_pairwise_inliers()