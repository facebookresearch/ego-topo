import argparse
import glob
import numpy as np
import os
import time
import cv2
import torch
import colorsys

import tqdm
import itertools
import collections
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from PIL import Image
import random
from random import shuffle

from utils import util
from data import epic, gtea


import argparse
parser = argparse.ArgumentParser(description='PyTorch SuperPoint Demo.')
parser.add_argument('--dset', default='epic')
args = parser.parse_args()

def get_descriptor(uid):
    # we have 16 of them per video clip, but just pick the middle one
    descriptor_data = torch.load(f'build_graph/data/{args.dset}/descriptors/{uid}.pth')
    descriptors = descriptor_data['desc']
    desc = descriptors[len(descriptors)//2] 
    frame = descriptor_data['frames'][len(descriptor_data['frames'])//2]
    return (desc['desc'], desc['pts'], frame)


def drawMatches(img1, kp1, img2, kp2, matches, inliers = None):

    # Create a new output image that concatenates the two images together
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx, img2_idx, score = mat
        img1_idx, img2_idx = int(img1_idx), int(img2_idx)

        # x - columns, y - rows
        (x1,y1) = kp1[img1_idx][:2]
        (x2,y2) = kp2[img2_idx][:2]

        inlier = False

        if inliers is not None:
            for i in inliers:
                if i[0] == x1 and i[1] == y1 and i[2] == x2 and i[3] == y2:
                    inlier = True

        # Draw a small circle at both co-ordinates
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points, draw inliers if we have them
        if inliers is not None and inlier:
            cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (0, 255, 0), 1)
        elif inliers is not None:
            cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (0, 0, 255), 1)

        if inliers is None:
            cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)

    return out

if __name__=='__main__':

    split = 'train'
    kitchen = 'P01'

    dset = epic.EPICInteractions('data/epic', split, 32)
    uid_to_entry = {entry['uid']:entry for entry in dset.data}

    neighbors = torch.load(f'build_graph/data/{args.dset}/matches/{split}/{kitchen}.pth')
    uids = neighbors['uids']
    homographies = neighbors['homography']
    entries = [uid_to_entry[uid] for uid in uids]

    neighbors = collections.defaultdict(list)
    for idx, (i,j) in enumerate(itertools.combinations(range(len(uids)), 2)):
        uid1, uid2 = uids[i], uids[j]
        homography = homographies[idx]

        if homography['inliers'] is None:
            continue

        neighbors[uid1].append({'nbh':uid2, 'matches':homography['matches'], 'inliers':homography['inliers']})
        neighbors[uid2].append({'nbh':uid1, 'matches':homography['matches'][[1, 0, 2]], 'inliers':homography['inliers'][:, [2, 3, 0, 1]]})

    shuffle(entries)
    for entry in entries:

        if len(neighbors[entry['uid']])==0:
            continue

        _, pts1, frame1 = get_descriptor(entry['uid'])

        nbhs = sorted(neighbors[entry['uid']], key=lambda x: -len(x['inliers']))
        nbhs = [nbh for nbh in nbhs if len(nbh['inliers'])>5]

        if len(nbhs)<1:
            continue

        # draw line matches
        nbh = nbhs[0]
        entry2 = uid_to_entry[nbh['nbh']]
        _, pts2, frame2 = get_descriptor(entry2['uid'])
        img1 = cv2.imread(dset.frame_path(frame1), 0)
        img2 = cv2.imread(dset.frame_path(frame2), 0)
        inlier_viz = drawMatches(img1, pts1.T, img2, pts2.T, nbh['matches'].T, nbh['inliers'])
        cv2.imshow('inliers', inlier_viz)


        # draw points on top neighbors
        img1 = cv2.imread(dset.frame_path(frame1))

        shuffle(nbhs)
        nbhs = nbhs[0:5]

        common_pts = set()
        nbh_imgs = []
        for nbh in nbhs:

            entry2 = uid_to_entry[nbh['nbh']]

            _, _, frame2 = get_descriptor(entry2['uid'])
            img2 = cv2.imread(dset.frame_path(frame2))

            m_pts1, m_pts2 = list(map(tuple, nbh['inliers'][:, :2])), list(map(tuple, nbh['inliers'][:, 2:]))
            common_pts |= set(m_pts1)

            match_dict = dict(zip(m_pts2, m_pts1))
            nbh_imgs.append({'img':img2, 'matches':match_dict})

        common_pts = list(common_pts)
        N = len(common_pts)
        HSV_tuples = [(x*1.0/N, 1, 1) for x in range(N)]
        colors = [tuple([int(y*255) for y in colorsys.hsv_to_rgb(*x)]) for x in  HSV_tuples]
        color_map = dict(zip(common_pts, colors))

        for pt in common_pts:
            cv2.circle(img1, pt, 4, color_map[pt], -1, lineType=cv2.LINE_AA)

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img1 = Image.fromarray(img1).resize((256, 256))
        img1 = transforms.ToTensor()(img1)
        img1 = util.add_border(img1, 'green', 256)

        viz_imgs = [img1]
        for nbh in nbh_imgs:
            img2 = nbh['img']

            print (nbh['matches'])

            for m_pt2, m_pt1 in nbh['matches'].items():
                cv2.circle(img2, m_pt2, 4, color_map[m_pt1], -1, lineType=cv2.LINE_AA)

            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            img2 = Image.fromarray(img2).resize((256, 256))
            img2 = transforms.ToTensor()(img2)
            viz_imgs.append(img2)

        grid = make_grid(viz_imgs, nrow=len(viz_imgs))
        util.show_wait(grid, T=0)



