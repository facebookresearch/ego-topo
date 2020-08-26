import numpy as np
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import itertools
import tqdm
import collections
import sys
import scipy.stats
from functools import lru_cache
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image, ImageOps, ImageDraw, ImageFont
import cv2
import copy
import json
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

import matplotlib
matplotlib.use('Agg')

from .utils import util
from .data import epic, gtea
from .localization_network.model import SiameseR18_5MLP, R18_5MLP

parser = argparse.ArgumentParser()
parser.add_argument('--v_id', default='P01_01', help='video or kitchen ID')
parser.add_argument('--dset', default=None, help='epic|gtea')
parser.add_argument('--cv_dir', default='cv/tmp/')
parser.add_argument('--viz', action='store_true', help='show graph construction over time')
parser.add_argument('--skip', type=int, default=1, help='number of frames to skip during viz')
parser.add_argument('--online', action='store_true', help='calculate frame scores online (vs. using precomputed score matrix)')
parser.add_argument('--frame_inp', action='store_true', help='use image inputs (vs. precomputed trunk features')
parser.add_argument('--locnet_wts', default='build_graph/localization_network/cv/dset/ckpt_E_?_A_?.pth', help='localization network weights to load')

args = parser.parse_args()

#-------------------------------------------------------------------------------------------------------------------#

def vprint(*pargs, **kwargs):
    if args.viz:
        print(*pargs, **kwargs)

def fig2tensor(fig): 
    fig.canvas.draw()
    data = np.array(fig.canvas.renderer._renderer)[:,:,:-1]
    data = torch.from_numpy(data).permute(2, 0, 1).float()/255.0
    return data

def resize_tensor(tensor, size):
    img = transforms.ToPILImage()(tensor)
    img = img.resize((size, size))
    img = transforms.ToTensor()(img)
    return img

#------------------------------------------------------------------------------------------------------------------#

class EnvGraph:

    def __init__(self, args):

        self.args = args

        if args.dset=='epic':
            self.dset = epic.EPICInteractions('build_graph/data/epic', 'val', 32)
            self.dset.fps = 60
            self.start_frame = 1
            self.end_frame = self.dset.annotations['vid_lengths'][args.v_id]
        elif args.dset=='gtea':
            self.dset = gtea.GTEAInteractions('build_graph/data/gtea', 'val', 32)
            self.dset.fps = 24

            # a lot of GTEA are blank
            video_bounds = json.load(open('build_graph/data/gtea/video_start_end.json'))
            self.start_frame, self.end_frame = video_bounds[args.v_id]

        self.dset.transform = util.default_transform('val')

        # subsample frames to 6fps for graph generation
        vid_length = self.dset.annotations['vid_lengths'][args.v_id]
        subsample =  self.dset.fps//6
        self.frames = [(args.v_id, f_id) for f_id in range(1, vid_length+1, subsample)]
        self.frame_to_idx = {frame: idx for idx, frame in enumerate(self.frames)}

        print (f'Generating graph for {args.dset}-{args.v_id}. {len(self.frames)} frames')

        #-------------------------------------------------------------------------------------#

        '''
        Set up the graph:
        - leaders: the graph node that each frame belongs to
        - pos: the graph layout for visualization
        - last state: used to decide edges etc.
        - G: the nx Graph. Each node has a bunch of attributes:
            - visitation: a dict with start/end times of a visit to that node
            - img: the node thumbnail (most recent visitation)
        '''
        self.G = nx.Graph()
        self.G.pos = None
        self.state = {'frame':None, 'node':None, 'inactive':True, 'viz_node':self.start_frame, 'last_node':self.start_frame, 'phase':'localize'}
        self.create_new_node(self.G, self.start_frame, [{'start':self.start_frame, 'stop':self.start_frame}]) # temporary node until initialized

        #-------------------------------------------------------------------------------------#

        # number of frames after which the edge will be declared far
        # edge will still be made, but with attr color='grey' as opposed to 'black'
        self.edge_delay = 20
        self.thresh_upper = 0.7
        self.thresh_lower = 0.4    

        # Only recognize a node if there are N consecutive localizations there
        self.window_size = 9
        self.reset_buffer()
        self.history = []

        #-------------------------------------------------------------------------------------#

    def reset_buffer(self):
        self.buffer = {'start':None, 'stop':None, 'node':None}

    def score_pair_sets(self, set1, set2):

        if len(set1)==0 or len(set2)==0:
            return -1

        S = []
        for i in set1:
            score = np.mean([self.pair_scores(self.frames[i], self.frames[j]) for j in set2])
            S.append(score)
        S = np.mean(S)
        return S


    # compute the score of every node in the graph with a new frame_i
    # use the center frame for each visitation for each node to compute score
    def score_nodes(self, frame_i):
        scores = []
        for node in self.G:

            visits = self.G.nodes[node]['members']

            # 20 uniformly sampled visits 
            visits = [visits[idx] for idx in np.round(np.linspace(0, len(visits) - 1, 20)).astype(int)]

            key_frames = []
            for visit in visits: 
                frames = list(range(visit['start'], visit['stop']+1))
                if len(frames)<self.window_size:
                    key_frames += frames
                else:
                    mid = len(frames)//2
                    key_frames += frames[mid-self.window_size//2:mid+self.window_size//2] # center frames of the visit

            window = list(range(frame_i - self.window_size//2, frame_i+self.window_size//2+1))
            score = self.score_pair_sets(window, key_frames)
            scores.append({'node':node, 'score':score})
        return scores

    #------------------------------------------------------------------------------------------------------------#

    def create_new_node(self, G, frame, members=[]):
        G.add_node(frame, members=members)
        vprint ('NODE: %d'%frame)

    def create_new_edge(self, G, frame, src, dst):

        if src is None or dst is None or src==dst:
            return

        last_node = self.state['last_node']
        if last_node is not None:
            last_visit = G.nodes[last_node]['members'][-1]
            last_frame = last_visit['stop']
        else:
            last_frame = 0

        dT = frame - last_frame
        if G.has_edge(src, dst):
            G[src][dst]['dT'].append(dT)
        else:
            G.add_edge(src, dst, dT=[dT])
        vprint ('EDGE: %d --> %d | T: %d'%(src, dst, dT))


    def score_state(self, frame_i):
        node_scores = self.score_nodes(frame_i)
        node_scores = sorted(node_scores, key=lambda node: -node['score'])

        top1_node = node_scores[0]
        top2_node = node_scores[1] if len(node_scores)>1 else {'score':0}

        trigger = 'skip'
        if top1_node['score']>self.thresh_upper and top1_node['score']-top2_node['score']>0.1:
            trigger = 'localize node'
        elif top1_node['score']<self.thresh_lower:
            trigger = 'create node'

        return top1_node, top2_node, trigger, node_scores


    def create_step(self, frame_i):

        best_node, best_node2, trigger, _ = self.score_state(frame_i)

        if 'create_count' not in self.buffer:
            self.reset_buffer()
            self.buffer['create_count'] = 0
            self.buffer['start'] = frame_i
            self.buffer['stop'] = frame_i

        # still wants to create a new node
        if trigger=='create node':

            if self.buffer['create_count']<5:
                self.buffer['create_count'] += 1
                self.buffer['stop'] = frame_i
                return None, None

            else:
                node_i = self.buffer['start']
                visit = {'start':self.buffer['start'], 'stop':self.buffer['stop'], 'node':node_i}
                self.create_new_node(self.G, node_i, members=[visit])
                self.create_new_edge(self.G, frame_i, self.state['last_node'], node_i)

                self.state['last_node'] = node_i
                self.state['phase'] = 'localize'
                self.buffer = visit
            return node_i, 1.0

        # No new node is created. Back to localization cycle
        elif trigger=='localize node':
            self.reset_buffer()
            return self.localize_step(frame_i)

        elif trigger=='skip':
            self.reset_buffer()
            self.state['phase'] = 'localize'
    
        return None, None

    def localize_step(self, frame_i):

        best_node, best_node2, trigger, _ = self.score_state(frame_i)

        if trigger=='localize node':

            node_i, score_i = best_node['node'], best_node['score']

            if self.buffer['node']==node_i:
                self.buffer['stop'] = frame_i
            else:
                visit = {'start':frame_i, 'stop':frame_i, 'node':node_i}
                self.G.nodes[node_i]['members'].append(visit)
                self.buffer = visit
                self.create_new_edge(self.G, frame_i, self.state['last_node'], node_i)

            self.state['phase'] = 'localize'
            self.state['last_node'] = node_i
            return node_i, score_i

        elif trigger=='create node':
            self.state['phase'] = 'create'
            return None, None

        elif trigger=='skip':
            self.state['phase'] = 'localize'
            return None, None

    def print_state(self, frame_i):

        best_node, best_node2, trigger, node_scores = self.score_state(frame_i)
        vprint ('buffer:', self.buffer)
        vprint ('phase:', self.state['phase'])

        for item in sorted(node_scores, key=lambda x: x['node']):
            prefix = '*' if best_node['node']==item['node'] else ''
            visits = self.G.nodes[item['node']]['members']
            visits = ', '.join(['%s-->%s'%(visit['start'], visit['stop']) for visit in visits])
            vprint ('%s%s: %.3f | %s'%(prefix, item['node'], item['score'], visits))
        vprint ('-'*20)


    def log_history(self, frame_i):
        self.history.append({'frame': frame_i, 'G':copy.deepcopy(self.G), 'state':dict(self.state)})

    def build(self):

        last_vid = self.frames[0][0]
        for frame_i in tqdm.tqdm(range(self.window_size//2, len(self.frames)-self.window_size//2)):

            if self.frames[frame_i][1] < self.start_frame or self.frames[frame_i][1] > self.end_frame:
                continue

            # for combined graphs, if we move to the next video, refresh everything
            v_id = self.frames[frame_i][0]
            if v_id!=last_vid:
                self.reset_buffer()
                self.state['last_node'] = None
                print ('Moving from %s --> %s. Clearing buffers.'%(last_vid, v_id))
                last_vid = v_id

            if self.state['phase']=='localize':
                node_i, score_i = self.localize_step(frame_i)
            elif self.state['phase']=='create':
                node_i, score_i = self.create_step(frame_i)

            self.print_state(frame_i)
            self.state.update({'frame':frame_i, 'inactive':self.state['node'] is None, 'node':node_i, 'score':score_i, 'viz_node':node_i or self.state['viz_node']})
            self.log_history(frame_i)

            if frame_i%self.args.skip==0:
                self.viz(frame_i, block_size=256, img_sz=50, nsz=25, out_sz=1024)

        self.save_history() 


    def load_frame(self, frame):
        frame = self.dset.load_image(frame)
        frame = self.dset.transform(frame)
        return frame   

    def load_frame_viz(self, frame):
        frame = self.load_frame(frame)
        frame = util.unnormalize(frame, 'imagenet')
        return frame   
        

    def viz(self, i, block_size=512, img_sz=100, nsz=50, out_sz=2048):

        if not self.args.viz:
            return
        viz_items = []


        # current frame
        frame = self.load_frame_viz(self.frames[self.state['frame']])
        frame = resize_tensor(frame, block_size)
        viz_items.append(frame)

        #-----------------------------------------------------------------#

        # visited state grid
        frame_imgs = {}
        nbhs = set(self.G.neighbors(self.state['viz_node'])) if self.state['viz_node'] in self.G else set()
        node_ref = []
        for n in self.G.nodes():

            last_visit = self.G.nodes[n]['members'][-1]
            frame_i = (last_visit['start'] + last_visit['stop'])//2
            img = self.load_frame_viz(self.frames[frame_i])
            img = transforms.ToPILImage()(img)

            if n==self.state['viz_node']:
                color = 'grey' if self.state['inactive'] else 'green'
                img = ImageOps.expand(img, border=15, fill=color)
            elif n in nbhs and not self.state['inactive']:
                img = ImageOps.expand(img, border=15, fill='blue')
            
            img = img.resize((img_sz, img_sz))

            node_ref.append(img)
            frame_imgs[n] = img

        N = 5
        node_ref = node_ref[:N**2]
        node_ref = node_ref + [Image.new("RGB", (img_sz, img_sz), "white")]*(N**2-len(node_ref))
        node_ref = [transforms.ToTensor()(img) for img in node_ref]
        grid = make_grid(node_ref, nrow=N)
        grid = resize_tensor(grid, block_size)
        viz_items.append(grid)

        #-----------------------------------------------------------------#

        # graph
        plt.clf()
        fig = plt.gcf()
        fig.set_size_inches(block_size/fig.get_dpi(), block_size/fig.get_dpi())
        plt.xlim(-1,1)
        plt.ylim(-1,1)

        self.G.pos = nx.spring_layout(self.G, scale=0.8, pos=self.G.pos, iterations=5)
        edges = self.G.edges()
        colors = ['black' if np.min(self.G[u][v]['dT']) < self.edge_delay else 'grey' for u, v in edges]
        nx.draw(self.G, self.G.pos, edgelist=edges, edge_color=colors, node_size=10)

        for n in self.G.nodes():
            x, y = self.G.pos[n]
            x, y = int((x+1)/2*block_size), int((y+1)/2*block_size)
            img = frame_imgs[n].resize((nsz, nsz))
            fig.figimage(img, x-nsz//2, y-nsz//2, zorder=10)
        graph = fig2tensor(fig)
        graph = resize_tensor(graph, block_size)
        viz_items.append(graph)

        #-----------------------------------------------------------------#

        viz = make_grid(viz_items, nrow=3)
        util.show_wait(viz, sz=out_sz, T=0)

    def save_history(self):
        if not self.args.viz: 
            torch.save({'frames':self.frames, 'graphs':self.history}, f'{self.args.out_dir}/{self.args.v_id}_graph.pth')


#---------------------------------------------------------------------------------------------------------------------#

class EnvGraphOffline(EnvGraph):

    def __init__(self, args):

        super().__init__(args)
        print ('** NOTE ** Generating graph in OFFLINE mode')

        pairwise_score_file = f'build_graph/data/{args.dset}/locnet_pairwise_scores/{args.v_id}_scores.pth'
        scores = np.memmap(pairwise_score_file, dtype='float32', mode='r')
        score_matrix = np.zeros((len(self.frames), len(self.frames)))
        score_matrix[np.triu_indices_from(score_matrix, k=1)] = scores
        score_matrix = score_matrix + score_matrix.T
        score_matrix[np.diag_indices(len(self.frames))] = 1.0
        self.score_matrix = score_matrix
        print ('Loaded pair scores')

    def pair_scores(self, fA, fB):
        idxA, idxB = self.frame_to_idx[fA], self.frame_to_idx[fB]
        return self.score_matrix[idxA, idxB]


class EnvGraphOnline(EnvGraph):

    def __init__(self, args):

        super().__init__(args)
        print ('** NOTE ** Generating graph in ONLINE mode')

        if args.frame_inp:
            print ('** NOTE ** Using raw frames as input')
            self.net = SiameseR18_5MLP()
        else:
            print ('** NOTE ** Using precomputed localization net trunk features')
            self.net = R18_5MLP()
            self.feat_cache = torch.load(f'build_graph/data/{args.dset}/locnet_trunk_feats.pth')
            args.locnet_wts = os.path.join(os.path.dirname(args.locnet_wts), 'head.pth')

        checkpoint = torch.load(args.locnet_wts, map_location='cpu')
        self.net.load_state_dict(checkpoint['net'])
        self.net.cuda().eval()
        print (f'Loaded checkpoint from {os.path.basename(args.locnet_wts)}')


    # use a resnet to generate these scores
    @lru_cache(maxsize=1000000)
    def pair_scores(self, fA, fB):
        if fB < fA: # symmetry
            fA, fB = fB, fA

        if self.args.frame_inp:
            featA, featB = self.load_frame(fA), self.load_frame(fB)
        else:
            featA, featB = self.feat_cache[fA], self.feat_cache[fB]

        with torch.no_grad():
            sim = self.net({'imgA':featA.unsqueeze(0).cuda(), 'imgB':featB.unsqueeze(0).cuda()}, softmax=True)
        sim = sim[0].item()
        return sim



#-------------------------------------------------------------------------------------------------------------------#
if __name__ == '__main__':

    args.out_dir = f'build_graph/data/{args.dset}/graphs/'
    os.makedirs(args.out_dir, exist_ok=True)

    if args.online:
        env_graph = EnvGraphOnline(args)
    else:
        env_graph = EnvGraphOffline(args)        
    env_graph.build()

