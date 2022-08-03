# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from . import gtransforms


# Apply .cuda() to every element in the batch
def batch_cuda(batch):
    _batch = {}
    for k,v in batch.items():
        if type(v)==torch.Tensor:
            v = v.cuda()
        elif type(v)==list and type(v[0])==torch.Tensor:
            v = [v.cuda() for v in v]
        _batch.update({k:v})

    return _batch


# NOTE: Single channel mean/stev (unlike pytorch Imagenet)
def kinetics_mean_std():
    mean = [114.75, 114.75, 114.75]
    std = [57.375, 57.375, 57.375]
    return mean, std

def clip_transform(split, max_len):

    mean, std = kinetics_mean_std()
    if split=='train':
        transform = transforms.Compose([
                        gtransforms.GroupResize(256),
                        gtransforms.GroupRandomCrop(224),
                        gtransforms.GroupRandomHorizontalFlip(),
                        gtransforms.ToTensor(),
                        gtransforms.GroupNormalize(mean, std),
                        gtransforms.LoopPad(max_len),
                    ])

    elif split=='val':
        transform = transforms.Compose([
                        gtransforms.GroupResize(256),
                        gtransforms.GroupCenterCrop(256),
                        gtransforms.ToTensor(),
                        gtransforms.GroupNormalize(mean, std),
                        gtransforms.LoopPad(max_len),
            ])

    return transform

def unnormalize(tensor, mode='default'):
    mean, std = kinetics_mean_std() if mode=='kinetics' else default_mean_std()
    u_tensor = tensor.clone()

    def _unnorm(t):
        for c in range(3):
            t[c].mul_(std[c]).add_(mean[c])

    if u_tensor.dim()==4:
        [_unnorm(t) for t in u_tensor]
    else:
        _unnorm(u_tensor)
    
    return u_tensor

def default_mean_std():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return mean, std

def default_transform(split):
    mean, std = default_mean_std()

    if split=='train':
        transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.RandomCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std)
                    ])


    elif split=='val':
        transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std)
            ])

    elif split=='val224':
        transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
            ])


    return transform

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k)
    return res


import cv2
import sys
def show_wait(img, T=0, win='image', sz=None, save=None):

    shape = img.shape
    img = transforms.ToPILImage()(img)
    if sz is not None:
        H_new = int(sz/shape[2]*shape[1])
        img = img.resize((sz, H_new))

    open_cv_image = np.array(img) 
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    if save is not None:
        cv2.imwrite(save, open_cv_image)
        return

    cv2.imshow(win, open_cv_image)
    inp = cv2.waitKey(T)
    if inp==27:
        cv2.destroyAllWindows()
        sys.exit(0)

from PIL import Image, ImageOps
def add_border(img, color, sz=128):
    img = transforms.ToPILImage()(img)
    img = ImageOps.expand(img, border=5, fill=color)
    img = img.resize((sz, sz))
    img = transforms.ToTensor()(img)
    return img

