import os
import glob
import numpy as np
import cv2
import torch
import tqdm

import data
from data import epic, gtea
from .superpoint.model import SuperPointFrontend


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--chunk', type=int, default=None, help='current chunk')
parser.add_argument('--nchunks', type=int, default=None, help='total chunks')
parser.add_argument('--load', default='build_graph/tools/superpoint/superpoint_v1.pth')
parser.add_argument('--nms_dist', type=int, default=4)
parser.add_argument('--conf_thresh', type=float, default=0.015)
parser.add_argument('--nn_thresh', type=float, default=0.7)
parser.add_argument('--cv_dir', default=None)
parser.add_argument('--dset', default=None)
args = parser.parse_args()


def load_frame(dset, frame):
    file = dset.frame_path(frame)
    img = cv2.imread(file, 0)
    img = (img.astype('float32') / 255.)
    return img

def run():

    descriptor_dir = f'build_graph/data/{args.dset}/descriptors'
    os.makedirs(descriptor_dir, exist_ok=True)

    if args.dset=='epic':
        dset = epic.EPICInteractions('data/epic', 'val', 32)
    elif args.dset=='gtea':
        dset = gtea.GTEAInteractions('data/gtea', 'val', 32)

    # Gather 16 uniformly spaced frames from each clip to genereate descriptors for
    entries = dset.train_data + dset.val_data
    frame_list = []
    for entry in entries:
        frames = [entry['frames'][idx] for idx in np.round(np.linspace(0, len(entry['frames']) - 1, 16)).astype(int)] 
        frame_list.append({'uid':entry['uid'], 'frames':frames})

    # Split data into chunks and process. See .sh file
    nchunks = args.nchunks
    chunk_size = len(frame_list)//nchunks
    chunk_data = frame_list[args.chunk*chunk_size:args.chunk*chunk_size + chunk_size]

    # create the superpoint model and load weights
    fe = SuperPointFrontend(weights_path=args.load, nms_dist=args.nms_dist, conf_thresh=args.conf_thresh, nn_thresh=args.nn_thresh, cuda=True)

    # generate SP descriptors for these frames
    for entry in tqdm.tqdm(chunk_data, total=len(chunk_data)):

        descriptors = []
        for frame in entry['frames']:
            img = load_frame(dset, frame)
            pts, desc, _ = fe.run(img) # (3, N), (256, N)
            descriptors.append({'pts':pts, 'desc':desc})

        descriptors = {'frames':entry['frames'], 'desc':descriptors}
        torch.save(descriptors, f'{descriptor_dir}/{entry["uid"]}')


if __name__=='__main__':
    # call using .sh script to do this in parallel
    # ideally by submitting each run to cluster nodes
    # > bash generate_sp_descriptors.sh epic
    run()








