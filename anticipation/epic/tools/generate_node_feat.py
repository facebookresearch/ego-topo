import argparse

import collections
import os
import torch
import mmcv
import numpy as np
from mmcv.runner import load_checkpoint, parallel_test, obj_from_dict
from mmcv.parallel import scatter, collate, MMDataParallel

from mmaction import datasets
from mmaction.datasets import build_dataloader
from mmaction.models import build_recognizer, recognizers
from mmaction.core.evaluation.accuracy import (softmax, top_k_accuracy,
                                               mean_class_accuracy)
from mmaction.apis.env import get_root_logger

from .. import datasets
from ..models import *
import epic.utils as utils


def single_test(model, data_loader):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)
        results.append(result)

        batch_size = data['img_group_0'].data[0].size(0) \
            if 'img_group_0' in data else data['feature'].data[0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
        
        # print(result[:10])
    return results


def _data_func(data, device_id):
    data = scatter(collate([data], samples_per_gpu=1), [device_id])[0]
    return dict(return_loss=False, rescale=True, **data)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate lfb')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--input-dir', help="load checkpoint work dir")
    # parser.add_argument('anno_file', help='input anno file')
    parser.add_argument('--mode', default='test')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument(
        '--gpus', default=1, type=int, help='GPU number used for testing')
    parser.add_argument(
        '--proc_per_gpu',
        default=1,
        type=int,
        help='Number of processes per GPU')
    parser.add_argument('--save_path', help='output file')
    args = parser.parse_args()
    return args


def generate_node_feat(_cfg, checkpoint, mode, save_feat_path, gpus=1, logger=None):
    cfg = _cfg.copy()
    if save_feat_path is not None and not save_feat_path.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    if logger is None:
        logger = get_root_logger(cfg.log_level)

    # update test configs for lfb
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # cfg.model.reg_head = None

    data_cfg = cfg.data.get(mode)
    dataset = obj_from_dict(data_cfg, datasets, dict(test_mode=True))
    assert gpus == 1, "1 gpu is faster now"
    model = build_recognizer(
        cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    load_checkpoint(model, checkpoint, strict=False)
    model = MMDataParallel(model, device_ids=[0])

    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        num_gpus=1,
        dist=False,
        shuffle=False)
    outputs = single_test(model, data_loader)

    print(np.mean(outputs[0]))

    assert(len(outputs) == len(dataset.records))
    feature_bank = collections.defaultdict(dict)
    # class_info = collections.defaultdict(dict)
    for output, record in zip(outputs, dataset.records):
        v_path, t, node = record
        feat = output.reshape(-1)
        # vid = video_info.path
        # frame = "{}-{}".format(video_info.start_frame, video_info.end_frame)
        feature_bank[(v_path, t)][node] = feat
      
    feature_bank = dict(feature_bank)
   
    if save_feat_path is not None:
        logger.info('writing feat results to {}'.format(save_feat_path))
        mmcv.dump(feature_bank, save_feat_path)

def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    if args.input_dir is not None:
        cfg.work_dir = args.input_dir
    if args.checkpoint is None:
        args.checkpoint = os.path.join(cfg.work_dir, "latest.pth")
    if args.save_path is not None:
        args.save_path = os.path.join(cfg.work_dir, args.save_path)
    # if args.lfb_class_output is not None:
    #     args.lfb_class_output = os.path.join(cfg.work_dir, args.lfb_class_output)
    # clip_info = generate_clip_info(args.anno_file)
    generate_node_feat(
        cfg, args.checkpoint, mode=args.mode, save_feat_path=args.save_path
    )


if __name__ == '__main__':
    main()
