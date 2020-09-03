# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import argparse
import glob
import numpy as np
import os
import time
import cv2
import torch
import tqdm
import itertools
import collections
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from PIL import Image
import random
from random import shuffle
import functools
from joblib import Parallel, delayed

from ..utils import util
from ..data import epic, gtea

parser = argparse.ArgumentParser(description='PyTorch SuperPoint Demo.')
parser.add_argument('--chunk', default=None)
parser.add_argument('--split', default=None)
parser.add_argument('--nn_thresh', type=float, default=0.7)
parser.add_argument('--cv_dir', default=None)
parser.add_argument('--dset', default=None)
args = parser.parse_args()

def nn_match_two_way(desc1, desc2, nn_thresh):
    """
    Performs two-way nearest neighbor matching of two sets of descriptors, such
    that the NN match from descriptor A->B must equal the NN match from B->A.

    Inputs:
      desc1 - NxM numpy matrix of N corresponding M-dimensional descriptors.
      desc2 - NxM numpy matrix of N corresponding M-dimensional descriptors.
      nn_thresh - Optional descriptor distance below which is a good match.

    Returns:
      matches - 3xL numpy array, of L matches, where L <= N and each column i is
                a match of two descriptors, d_i in image 1 and d_j' in image 2:
                [d_i index, d_j' index, match_score]^T
    """
    assert desc1.shape[0] == desc2.shape[0]
    if desc1.shape[1] == 0 or desc2.shape[1] == 0:
      return np.zeros((3, 0))
    if nn_thresh < 0.0:
      raise ValueError('\'nn_thresh\' should be non-negative')
    # Compute L2 distance. Easy since vectors are unit normalized.
    dmat = np.dot(desc1.T, desc2)
    dmat = np.sqrt(2-2*np.clip(dmat, -1, 1))
    # Get NN indices and scores.
    idx = np.argmin(dmat, axis=1)
    scores = dmat[np.arange(dmat.shape[0]), idx]
    # Threshold the NN matches.
    keep = scores < nn_thresh
    # Check if nearest neighbor goes both directions and keep those.
    idx2 = np.argmin(dmat, axis=0)
    keep_bi = np.arange(len(idx)) == idx2[idx]
    keep = np.logical_and(keep, keep_bi)
    idx = idx[keep]
    scores = scores[keep]
    # Get the surviving point indices.
    m_idx1 = np.arange(desc1.shape[1])[keep]
    m_idx2 = idx
    # Populate the final 3xN match data structure.
    matches = np.zeros((3, int(keep.sum())))
    matches[0, :] = m_idx1
    matches[1, :] = m_idx2
    matches[2, :] = scores
    return matches

def geometricDistance(correspondence, h):

    p1 = np.transpose(np.matrix([correspondence[0], correspondence[1], 1]))
    estimatep2 = np.dot(h, p1)
    estimatep2 = (1/estimatep2.item(2))*estimatep2

    p2 = np.transpose(np.matrix([correspondence[2], correspondence[3], 1]))
    error = p2 - estimatep2
    return np.linalg.norm(error)

def calculateHomography(correspondences):
    #loop through correspondences and create assemble matrix
    aList = []
    for corr in correspondences:
        p1 = np.matrix([corr.item(0), corr.item(1), 1])
        p2 = np.matrix([corr.item(2), corr.item(3), 1])

        a2 = [0, 0, 0, -p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2),
              p2.item(1) * p1.item(0), p2.item(1) * p1.item(1), p2.item(1) * p1.item(2)]
        a1 = [-p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2), 0, 0, 0,
              p2.item(0) * p1.item(0), p2.item(0) * p1.item(1), p2.item(0) * p1.item(2)]
        aList.append(a1)
        aList.append(a2)

    matrixA = np.matrix(aList)

    #svd composition
    u, s, v = np.linalg.svd(matrixA)

    #reshape the min singular value into a 3 by 3 matrix
    h = np.reshape(v[8], (3, 3))

    #normalize and now we have h
    h = (1/h.item(8)) * h
    return h


def ransac(corr, thresh):
    maxInliers = []
    finalH = None
    for i in range(1000):

        pts = np.stack([corr[np.random.randint(len(corr))] for _ in range(4)], 0)
        h = calculateHomography(pts)
        inliers = []

        for i in range(len(corr)):
            d = geometricDistance(corr[i], h)
            if d < 5:
                inliers.append(corr[i])

        if len(inliers) > len(maxInliers):
            maxInliers = inliers
            finalH = h
        # print ("Corr size: ", len(corr), " NumInliers: ", len(inliers), "Max inliers: ", len(maxInliers))

        if len(maxInliers) > (len(corr)*thresh):
            break

    return finalH, np.array(maxInliers, dtype=np.int32)


def get_descriptor(uid):
    # we have 16 of them per video clip, but just pick the middle one
    descriptors = torch.load(f'build_graph/data/{args.dset}/descriptors/{uid}')['desc']
    desc = descriptors[len(descriptors)//2] 
    return (desc['desc'], desc['pts'])


def get_pair_homography(descrip1, descrip2):
    desc1, pts1 = descrip1
    desc2, pts2 = descrip2
    try:
        matches = nn_match_two_way(desc1, desc2, args.nn_thresh)
    except:
        matches = None

    if matches is None or matches.shape[1]<4:
        return {'matches':None, 'inliers':None, 'H':None}        

    corr = []
    for mat in matches.T:
        img1_idx, img2_idx, score = mat
        img1_idx, img2_idx = int(img1_idx), int(img2_idx)
        (x1,y1) = pts1[:2, img1_idx]
        (x2,y2) = pts2[:2, img2_idx]
        corr.append([x1, y1, x2, y2])
    corr = np.array(corr)

    try:
        finalH, inliers = ransac(corr, 0.60)
    except:
        finalH, inliers = None, None

    return {'matches':matches, 'inliers':inliers, 'H':finalH}        


def generate_matches(uids, match_dir):

    pairs = list(itertools.combinations(range(len(uids)), 2))
    print ('Generating homography for %d pairs'%len(pairs))
    descriptors = {uid: get_descriptor(uid) for uid in tqdm.tqdm(uids)}
    homography = Parallel(n_jobs=32, verbose=1)(delayed(get_pair_homography)(descriptors[uids[i]], descriptors[uids[j]]) for i, j in pairs)

    save_data = {'uids':uids, 'homography':homography}
    torch.save(save_data, f'{match_dir}/{args.chunk}.pth')


def run():

    random.seed(1234)
    match_dir = f'build_graph/data/{args.dset}/matches/{args.split}/'
    os.makedirs(match_dir, exist_ok=True)

    # Generate the list of clips to calculate matches across.
    # For epic: get matches across all clips in the same kitchen
    # For gtea: get matches across every clip
    if args.dset=='epic':
        # args.chunk should be the person ID (P01, P02 ...)
        # P22 is split into P22a + P22b because it has too many clips
        dset = epic.EPICInteractions('data/epic', args.split, 32)
        entries = [entry for entry in dset.data if args.chunk[0:3] in entry['uid']]

        uid_to_entry = {entry['uid']:entry for entry in entries}
        uids = sorted(uid_to_entry.keys())

        if args.chunk == 'P22a':
            shuffle(uids)
            uids = uids[0:2000]
        elif args.chunk == 'P22b':
            shuffle(uids)
            uids = uids[2000:]


    elif args.dset=='gtea':
        # args.chunk should be one of [train1, train2, train3, train4, val]
        dset = gtea.GTEAInteractions('data/gtea', args.split, 32)
        entries =  dset.data # all of them are in the same kitchen!

        uid_to_entry = {entry['uid']:entry for entry in entries}
        uids = sorted(uid_to_entry.keys())

        # train/val chunks only exists for their respective splits
        if args.split=='train' and args.chunk=='val':
            return
        if args.split=='val' and args.chunk!='val':
            return

        if args.split=='train':
            shuffle(uids)
            idx = int(args.chunk.split('train')[1])
            uids = uids[2069*(idx-1):2069*idx]


    # generate superpoint matches
    generate_matches(uids, match_dir)


if __name__=='__main__':

    run()






