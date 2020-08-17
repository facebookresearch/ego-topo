import argparse

import collections
import os
import torch
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


def single_test(model, data_loader):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        with torch.no_grad():
            result = model(return_loss=False, **data)
        results.append(result)

        batch_size = data['img_group_0'].data[0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def _data_func(data, device_id):
    data = scatter(collate([data], samples_per_gpu=1), [device_id])[0]
    return dict(return_loss=False, rescale=True, **data)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate lfb')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('video_list_path', help='input video list')
    parser.add_argument('anno_file', help='input anno file')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument(
        '--gpus', default=1, type=int, help='GPU number used for testing')
    parser.add_argument(
        '--proc_per_gpu',
        default=1,
        type=int,
        help='Number of processes per GPU')
    parser.add_argument('--lfb_clip_stride', type=int, default=60)
    parser.add_argument('--lfb_output', help='output lfb file')
    parser.add_argument('--lfb_class_output', help='output lfb class info file')
    args = parser.parse_args()
    return args


def generate_lfb(_cfg, video_list_path, save_lfb_path, save_class_path, clip_info, checkpoint, lfb_clip_stride=60, gpus=1, proc_per_gpu=1, logger=None):
    cfg = _cfg.copy()
    if save_lfb_path is not None and not save_lfb_path.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    if logger is None:
        logger = get_root_logger(cfg.log_level)

    # update test configs for lfb
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True
    cfg.data.test.oversample = None
    cfg.data.test.num_segments = 1
    cfg.data.test.ann_file = video_list_path
    cfg.data.workers_per_gpu = 16
    if cfg.data.test.input_size == 256:
        cfg.model.spatial_temporal_module.spatial_size = 8

    cfg.model.cls_head["return_feat"] = True

    dataset = obj_from_dict(cfg.data.test, datasets, dict(test_mode=True, lfb_infer=True, lfb_clip_stride=lfb_clip_stride))
    if gpus == 1:
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
    else:
        model_args = cfg.model.copy()
        model_args.update(train_cfg=None, test_cfg=cfg.test_cfg)
        model_type = getattr(recognizers, model_args.pop('type'))
        outputs = parallel_test(
            model_type,
            model_args,
            checkpoint,
            dataset,
            _data_func,
            range(gpus),
            workers_per_gpu=proc_per_gpu)

    assert(len(outputs) == len(dataset.video_infos))
    feature_bank = collections.defaultdict(dict)
    class_info = collections.defaultdict(dict)
    for output, video_info in zip(outputs, dataset.video_infos):
        cls_score, feat = output[:-1], output[-1]
        # print(cls_score[0].shape, feat.shape)
        vid = video_info.path
        frame = (video_info.start_frame + video_info.end_frame) // 2
        feature_bank[vid][frame] = feat.reshape(-1)
        class_info[vid][frame] = {
            "score": [x.reshape(-1) for x in cls_score],
            "label": get_label(clip_info[vid], frame)
        }
    feature_bank = dict(feature_bank)
    class_info = dict(class_info)

    if save_lfb_path is not None:
        logger.info('writing lfb results to {}'.format(save_lfb_path))
        mmcv.dump(feature_bank, save_lfb_path)
    if save_class_path is not None:
        logger.info('writing lfb class info to {}'.format(save_class_path))
        mmcv.dump(class_info, save_class_path)

def generate_clip_info(anno_file):
    clip_info = collections.defaultdict(list)
    for row in open(anno_file):
        row = row.strip().split('\t')
        clip_info[row[0]].append((int(row[1]), int(row[2]), int(row[3])))

    clip_info = dict(clip_info)
    for key in clip_info:
        clip_info[key] = sorted(clip_info[key], key=lambda x: x[0])

    return clip_info


def get_label(clip_info, frame):
    for clip in clip_info:
        if clip[0] > frame:
            break
        if clip[0] <= frame and clip[1] >= frame:
            return clip[2]

    return 0


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    if args.checkpoint is None:
        args.checkpoint = os.path.join(cfg.work_dir, "latest.pth")
    if args.lfb_output is not None:
        args.lfb_output = os.path.join(cfg.work_dir, args.lfb_output)
    if args.lfb_class_output is not None:
        args.lfb_class_output = os.path.join(cfg.work_dir, args.lfb_class_output)
    clip_info = generate_clip_info(args.anno_file)
    generate_lfb(
        cfg, args.video_list_path, args.lfb_output, args.lfb_class_output, clip_info, args.checkpoint,
        lfb_clip_stride=args.lfb_clip_stride, gpus=args.gpus, proc_per_gpu=args.proc_per_gpu)


if __name__ == '__main__':
    main()
