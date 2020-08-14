import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data
import torch.utils.data.sampler
import numpy as np
import argparse
import tqdm
import torchnet as tnt
import collections
from torch.utils.tensorboard import SummaryWriter

from utils import util
from .dataset import EPICPairs, GTEAPairs
from .model import SiameseR18_5MLP



cudnn.benchmark = True 
parser = argparse.ArgumentParser()
parser.add_argument('--dset', default=None, help='gtea|epic')
parser.add_argument('--batch_size', default=256, type=int, help='Batch size for training')
parser.add_argument('--lr', default=2e-4, type=float, help='initial learning rate')
parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight decay for SGD')
parser.add_argument('--cv_dir', default='build_graph/localization_network/cv/tmp',help='Directory for saving checkpoint models')
parser.add_argument('--load', default=None)
parser.add_argument('--print_every', default=10, type=int)
parser.add_argument('--max_iter', default=20000, type=int)
parser.add_argument('--parallel', action ='store_true', default=False)
parser.add_argument('--workers', type=int, default=8)
args = parser.parse_args()

os.makedirs(args.cv_dir, exist_ok=True)

def save(epoch, iteration, net, optimizer, metadata=''):
    print('Saving state, iter:', iteration)
    state_dict = net.state_dict() if not args.parallel else net.module.state_dict()
    optim_state = optimizer.state_dict()
    checkpoint = {'net':state_dict, 'optimizer':optim_state, 'args':args, 'iter': iteration}
    torch.save(checkpoint, '%s/ckpt_E_%d_I_%d%s.pth'%(args.cv_dir, epoch, iteration, metadata))

def train(iteration, trainloader, valloader, net, optimizer, writer):

    net.train()

    total_iters = len(trainloader)
    epoch = iteration//total_iters
    plot_every = int(0.1*len(trainloader))
    loss_meters = collections.defaultdict(lambda: tnt.meter.MovingAverageValueMeter(20))

    while iteration <= args.max_iter:

        for batch in trainloader:

            batch = util.batch_cuda(batch)
            pred, loss_dict = net(batch)

            loss_dict = {k:v.mean() for k,v in loss_dict.items()}
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, pred_idx = pred.max(1)
            correct = (pred_idx==batch['label']).float().sum()
            batch_acc = correct/pred.shape[0]
            loss_meters['bAcc'].add(batch_acc.item())

            for k, v in loss_dict.items():
                loss_meters[k].add(v.item())
            loss_meters['total_loss'].add(loss.item())

            if iteration%args.print_every==0:
                log_str = 'iter: %d (%d + %d/%d) | '%(iteration, epoch, iteration%total_iters, total_iters)
                log_str += ' | '.join(['%s: %.3f'%(k, v.value()[0]) for k,v in loss_meters.items()])
                print (log_str)

            if iteration%plot_every==0:
                for key in loss_meters:
                    writer.add_scalar('train/%s'%key, loss_meters[key].value()[0], int(100*iteration/total_iters))

            iteration += 1
        
        epoch += 1

        if epoch%10==0:
            with torch.no_grad():
                validate(epoch, iteration, valloader, net, optimizer, writer)

 
def validate(epoch, iteration, valloader, net, optimizer, writer):

    net.eval()

    correct, total = 0, 0
    loss_meters = collections.defaultdict(lambda: tnt.meter.AverageValueMeter())

    for num_passes in tqdm.tqdm(range(10)): # run over validation set 10 times

        for batch in tqdm.tqdm(valloader, total=len(valloader)):

            batch = util.batch_cuda(batch)
            pred, loss_dict = net(batch)

            loss_dict = {k:v.mean() for k,v in loss_dict.items() if v.numel()>0}
            loss = sum(loss_dict.values())

            for k, v in loss_dict.items():
                loss_meters[k].add(v.item())
            loss_meters['total_loss'].add(loss.item())

            _, pred_idx = pred.max(1)
            correct += (pred_idx==batch['label']).float().sum()
            total += pred.size(0)

    accuracy = 1.0*correct/total

    log_str = '(val) E: %d | iter: %d | A: %.3f | '%(epoch, iteration, accuracy)
    log_str += ' | '.join(['%s: %.3f'%(k, v.value()[0]) for k,v in loss_meters.items()])
    print (log_str)

    val_stats = '_L_%.3f_A_%.3f'%(loss_meters['total_loss'].value()[0], accuracy)
    save(epoch, iteration, net, optimizer, val_stats)

    writer.add_scalar('val/loss', loss_meters['total_loss'].value()[0], epoch)
    writer.add_scalar('val/accuracy', accuracy, epoch)

    net.train()


def run():

    writer = SummaryWriter('%s/tb.log'%args.cv_dir)

    if args.dset=='epic':
        Dset = EPICPairs
        root = 'data/epic'
    elif args.dset=='gtea':
        Dset = GTEAPairs
        root = 'data/gtea'

    trainset = Dset(root, 'train')
    valset = Dset(root, 'val')

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    net = SiameseR18_5MLP()
    net.cuda()

    optim_params = list(filter(lambda p: p.requires_grad, net.parameters()))
    print ('Optimizing %d paramters'%len(optim_params))
    optimizer = optim.Adam(optim_params, lr=args.lr, weight_decay=args.weight_decay)

    start_iter = 0
    if args.load:
        checkpoint = torch.load(args.load, map_location='cpu')
        start_iter = checkpoint['iter']
        net.load_state_dict(checkpoint['net'])
        print ('Loaded checkpoint from %s'%os.path.basename(args.load))

    if args.parallel:
        net = nn.DataParallel(net)
        net.cuda()

    train(start_iter, trainloader, valloader, net, optimizer, writer)


#----------------------------------------------------------------------------------------------------------------------------------------#

if __name__=='__main__':
    run()
