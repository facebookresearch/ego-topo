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

from .. import datasets
from ..models import *
import epic.utils as utils


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
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--use_softmax', action='store_true',
                        help='whether to use softmax score')
    args = parser.parse_args()
    return args


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
    return results


def evaluate(results, labels, task):
    top1, top5 = top_k_accuracy(results, labels, k=(1, 5))
    mean_acc = mean_class_accuracy(results, labels)


    results = np.array(results)
    labels = np.array(labels)
    many_shot = pd.read_csv(
        "data/epic/annotations/EPIC_many_shot_{}s.csv".format(task)
    )['{}_class'.format(task)].values

    top5_recall = utils.topk_recall(
        results, labels, k=5, classes=many_shot)
   # tta_score = utils.tta(results.reshape(results[0], 1, -1), labels)

    # metrics = (mean_acc, top1, top5, top5_recall, tta_score)

    return mean_acc, top1, top5, top5_recall


def do_test(cfg, out, checkpoint, gpus=1, proc_per_gpu=1, use_softmax=False, task='verb', logger=None):
    if out is not None and not out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    if logger is None:
        # logger = get_root_logger(cfg.log_level)
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s', level=cfg.log_level)
        logger = logging.getLogger(__name__)
        filename = 'test.log'
        fh = logging.FileHandler(filename=os.path.join(cfg.work_dir, filename))
        fh.setLevel(cfg.log_level)
        logger.addHandler(fh)


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
   
    # support multi outputs
    if not isinstance(outputs[0], list):
        outputs = [[x] for x in outputs]

    gt_labels = []
    for i in range(len(dataset)):
        ann = dataset.get_ann_info(i)
        gt_labels.append(ann['label'])

    if use_softmax:
        logger.info("Averaging score over {} clips with softmax".format(
            outputs[0][0].shape[0]))
        results = [[softmax(x, dim=1).mean(axis=0) for x in res] for res in outputs]
    else:
        logger.info("Averaging score over {} clips without softmax (ie, raw)".format(
            outputs[0][0].shape[0]))
        results = [[x.mean(axis=0) for x in res] for res in outputs]
    metrics = []
    for i in range(len(results[0])):
        results_i = [res[i] for res in results]
        labels_i = [label[i] for label in gt_labels]

        mean_acc, top1, top5, top5_recall = evaluate(results_i, labels_i, task)
        logger.info("Task {} Mean Class Accuracy = {:.02f}".format(i, mean_acc * 100))
        logger.info("Task {} Top-1 Accuracy = {:.02f}".format(i, top1 * 100))
        logger.info("Task {} Top-5 Accuracy = {:.02f}".format(i, top5 * 100))
        logger.info("Task {} Top-5 Recall = {:.02f}".format(i, top5_recall * 100))
        # logger.info("Task {} tta score = {:.02f}".format(i, tta_score))
        
        metrics.append((mean_acc, top1, top5))

    if out is not None:
        logger.info('writing results to {}'.format(out))
        rets = {
            'outputs': outputs,
            'metrics': metrics,
        }
        mmcv.dump(rets, out)


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    if args.checkpoint is None:
        args.checkpoint = os.path.join(cfg.work_dir, "latest.pth")
    print("Testing on {}.".format(args.config))
    print("Checkpoint {}.".format(args.checkpoint))
    output_file = None
    if args.out is not None:
        output_file = args.out
    elif "output_file" in cfg:
        output_file = cfg.output_file
    do_test(cfg, output_file, args.checkpoint, args.gpus, args.proc_per_gpu, args.use_softmax, args.task)


if __name__ == '__main__':
    main()
