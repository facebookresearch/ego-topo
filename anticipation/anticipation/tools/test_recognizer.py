import argparse

import logging
import os
import torch
import pandas as pd
import numpy as np
import mmcv
from mmcv.runner import load_checkpoint, parallel_test, obj_from_dict
from mmcv.parallel import scatter, collate, MMDataParallel

from mmaction.datasets import build_dataloader
from mmaction.models import build_recognizer, recognizers
from mmaction.core.evaluation.accuracy import (softmax, top_k_accuracy,
                                               mean_class_accuracy)
from mmaction.apis.env import get_root_logger
from mmaction.apis.train import batch_processor

from .. import datasets
from ..models import *
import anticipation.utils as utils
from anticipation.runner.runner import LogBuffer


def parse_args():
    parser = argparse.ArgumentParser(description='Test an action recognizer')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--task', default='verb', type=str)
    parser.add_argument(
        '--gpus', default=1, type=int, help='GPU number used for testing')
    parser.add_argument(
        '--proc_per_gpu',
        default=1,
        type=int,
        help='Number of processes per GPU')
    args = parser.parse_args()
    return args


def single_test(model, data_loader):
    model.eval()
    # results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    log_buffer = LogBuffer()
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            outputs = batch_processor(model, data, train_mode=False)
        
        if 'log_vars' in outputs:
            log_buffer.update(outputs['log_vars'], outputs['num_samples'])

        batch_size = data['img_group_0'].data[0].size(0) \
            if 'img_group_0' in data else data['lfb'].data[0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    log_buffer.average()
    log_dict = {}
    for k, v in log_buffer.output.items():
        if 'mAP' in k:
            log_dict[k] = v

    return log_dict


def do_test(cfg, checkpoint, gpus=1, proc_per_gpu=1, task='verb', logger=None):
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True
    cfg.data.workers_per_gpu = 16
    # if 'input_size' in cfg.data.test and cfg.data.test.input_size == 256:
    #     cfg.model.spatial_temporal_module.spatial_size = 8

    dataset = obj_from_dict(cfg.data.test, datasets, dict(test_mode=True))
    assert gpus == 1, "1 gpu is faster now"
    model = build_recognizer(
        cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    load_checkpoint(model, checkpoint, strict=True)
    model = MMDataParallel(model, device_ids=[0])

    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        num_gpus=1,
        dist=False,
        shuffle=False)
    outputs = single_test(model, data_loader)

    print("\n---------------")
    print(outputs)


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    if args.checkpoint is None:
        args.checkpoint = os.path.join(cfg.work_dir, "latest.pth")
    print("Testing on {}.".format(args.config))
    print("Checkpoint {}.".format(args.checkpoint))
    do_test(cfg, args.checkpoint, args.gpus, args.proc_per_gpu, args.task)


if __name__ == '__main__':
    main()
