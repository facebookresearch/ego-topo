from __future__ import division

import argparse

import os
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

from mmaction import __version__
from mmaction.apis import (get_root_logger, init_dist, set_random_seed,
                           train_network)
from mmaction.models import build_recognizer
from mmcv import Config

from epic.datasets import get_trimmed_dataset
from epic.models import *


def parse_args():
    parser = argparse.ArgumentParser(description='Train an action recognizer')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus
    if cfg.checkpoint_config is not None:
        # save mmaction version in checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmact_version=__version__, config=cfg.text)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    model = build_recognizer(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    train_dataset = get_trimmed_dataset(cfg.data.train)
    val_dataset = get_trimmed_dataset(cfg.data.val)
    datasets = []
    for flow in cfg.workflow:
        assert flow[0] in ['train', 'val']
        if flow[0] == 'train':
            datasets.append(train_dataset)
        else:
            datasets.append(val_dataset)
    train_network(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=args.validate,
        logger=logger)
    
    from epic.tools.test_recognizer import do_test
    checkpoint = os.path.join(cfg.work_dir, "latest.pth")
    print("Testing on {}.".format(args.config))
    print("Checkpoint {}.".format(checkpoint))
    output_file = None
    if "output_file" in cfg:
        output_file = cfg.output_file
    do_test(cfg, output_file, checkpoint, 1)


if __name__ == '__main__':
    main()
