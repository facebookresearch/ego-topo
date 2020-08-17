import os.path as osp

import dgl
import numpy as np
import torch
from torch.utils.data import Dataset
import collections
import tqdm

import mmcv
from mmaction.datasets.transforms import GroupImageTransform
from mmcv.parallel import DataContainer as DC

from .epic_utils import EpicRawFramesRecord, to_tensor


class EpicRawFramesPretrainDataset(Dataset):
    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_norm_cfg,
                 num_segments=3,
                 new_length=1,
                 new_step=1,
                 random_shift=True,
                 temporal_jitter=False,
                 modality='RGB',
                 image_tmpl='img_{}.jpg',
                 img_scale=256,
                 img_scale_file=None,
                 input_size=224,
                 div_255=False,
                 size_divisor=None,
                 proposal_file=None,
                 num_max_proposals=1000,
                 flip_ratio=0.5,
                 resize_keep_ratio=True,
                 resize_ratio=[1, 0.875, 0.75, 0.66],
                 test_mode=False,
                 oversample=None,
                 random_crop=False,
                 more_fix_crop=False,
                 multiscale_crop=False,
                 scales=None,
                 max_distort=1,
                 input_format='NCHW',
                 anticipation_task=False,
                 anti_ta=60,
                 anti_to=64,
                 lfb_infer=False,
                 lfb_clip_stride=60):
        # prefix of images path
        self.img_prefix = img_prefix

        # normalization config
        self.img_norm_cfg = img_norm_cfg

        # parameters for frame fetching
        # number of segments
        self.num_segments = num_segments
        # number of consecutive frames
        self.old_length = new_length * new_step
        self.new_length = new_length
        # number of steps (sparse sampling for efficiency of io)
        self.new_step = new_step
        # whether to temporally random shift when training
        self.random_shift = random_shift
        # whether to temporally jitter if new_step > 1
        self.temporal_jitter = temporal_jitter

        # if used for lfb inference
        self.lfb_infer = lfb_infer
        # lfb clip stride (have a clip per stride)
        self.lfb_clip_stride = lfb_clip_stride
        # whether is anticipation task
        self.anticipation_task = anticipation_task
        self.anti_ta = anti_ta
        self.anti_to = anti_to

        self.test_mode = test_mode


        video_infos = [EpicRawFramesRecord(x.strip().split('\t')) for x in open(ann_file)]
        frame_to_record = {}
        int_counts = []
        for record in video_infos:
            record.v_id = record.path.split('/')[1]
            for f_id in range(record.start_frame, record.end_frame+1):
                frame_to_record[(record.v_id, f_id)] = record
            record.uid = '%s_%s_%s'%(record.path, record.start_frame, record.end_frame)
            int_counts.append((record.label[0], record.label[1]))
        self.frame_to_record = frame_to_record

        int_counts = collections.Counter(int_counts).items()
        int_counts = sorted(int_counts, key=lambda x: -x[1])[0:250]
        self.int_to_idx = {interact:idx for idx, (interact, count) in enumerate(int_counts)}


        # load annotations
        self.video_infos = self.load_annotations(ann_file)

        print ('Kept %d'%len(self.video_infos))



        # parameters for modalities
        if isinstance(modality, (list, tuple)):
            self.modalities = modality
            num_modality = len(modality)
        else:
            self.modalities = [modality]
            num_modality = 1
        if isinstance(image_tmpl, (list, tuple)):
            self.image_tmpls = image_tmpl
        else:
            self.image_tmpls = [image_tmpl]
        assert len(self.image_tmpls) == num_modality

        # parameters for image preprocessing
        # img_scale
        if isinstance(img_scale, int):
            img_scale = (np.Inf, img_scale)
        self.img_scale = img_scale
        if img_scale_file is not None:
            self.img_scale_dict = {line.split(' ')[0]:
                                   (int(line.split(' ')[1]),
                                    int(line.split(' ')[2]))
                                   for line in open(img_scale_file)}
        else:
            self.img_scale_dict = None
        # network input size
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.input_size = input_size

        # parameters for specification from pre-trained networks (lecacy issue)
        self.div_255 = div_255

        # parameters for data augmentation
        # flip ratio
        self.flip_ratio = flip_ratio
        self.resize_keep_ratio = resize_keep_ratio

        # test mode or not
        self.test_mode = test_mode

        self.flag = np.ones(len(self), dtype=np.uint8)

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

        # transforms
        assert oversample in [None, 'three_crop', 'ten_crop']
        self.img_group_transform = GroupImageTransform(
            size_divisor=None, crop_size=self.input_size,
            oversample=oversample, random_crop=random_crop,
            more_fix_crop=more_fix_crop,
            multiscale_crop=multiscale_crop, scales=scales,
            max_distort=max_distort,
            **self.img_norm_cfg)

        # input format
        assert input_format in ['NCHW', 'NCTHW']
        self.input_format = input_format
        '''
        self.bbox_transform = Bbox_transform()
        '''

    def __len__(self):
        return len(self.video_infos)


    def visits_to_intdist(self, visits):
        v_id = visits[0]['start'][0]

        n_frames = []
        for visit in visits:
            n_frames += [(v_id, f_id) for f_id in range(visit['start'][1], visit['stop'][1]+1)]

        records = [self.frame_to_record[frame] for frame in n_frames if frame in self.frame_to_record]
        records = {record.uid:record for record in records}.values() # remove duplicate entries

        def get_dist(recs, N, label_fn):
            counts = []
            for record in recs:
                counts.append(label_fn(record))
            counts = collections.Counter(counts)

            dist = torch.zeros(N)
            if len(counts)>0:
                for item, count in counts.items():
                    dist[item] = count
                dist = dist/dist.sum()
            return dist

        verb_dist = get_dist(records, 125, lambda record: record.label[0])
        noun_dist = get_dist(records, 352, lambda record: record.label[1])
        # int_dist = get_dist([record for record in records if (record.label[0], record.label[1]) in self.int_to_idx], 250, lambda record: self.int_to_idx[(record.label[0], record.label[1])])

        return verb_dist, noun_dist#, int_dist


    def load_annotations(self, ann_file):
        if self.lfb_infer:
            clips = []
            for x in open(ann_file):
                video_path, num_frame = x.strip().split('\t')
                for i in range(self.old_length // 2, int(num_frame) - self.old_length // 2 + 1, self.lfb_clip_stride):
                    clips.append(EpicRawFramesRecord(
                        [video_path, i - self.old_length // 2, i + self.old_length // 2, 0]
                    ))
            return clips
        else:

            phase = 'val' if self.test_mode else 'train'
            entries = torch.load('data/universal_graph_data.pth')['%s_data'%phase]
            print ('loaded %s data'%phase.upper())

            records = []
            for entry in entries:

                if entry['start_frame'][0]!=entry['stop_frame'][0]:
                    continue

                v_id = entry['start_frame'][0]
                path = '%s/%s'%(v_id.split('_')[0], v_id)
                data = [path, entry['start_frame'][1], entry['stop_frame'][1], -1, -1]
                visits = entry['visits']
                for visit in visits:
                    visit['start'] = visit['start_frame']
                    visit['stop'] = visit['stop_frame']

                record = EpicRawFramesRecord(data)
                record.visits = visits
                record.node = entry['node']

                records.append(record)

            print ('%d records loaded!!!'%len(records))

            return records
        # return mmcv.load(ann_file)


    def load_proposals(self, proposal_file):
        return mmcv.load(proposal_file)

    def get_ann_info(self, idx):
        return {'path': self.video_infos[idx].path,
                'num_frames': self.video_infos[idx].num_frames,
                'label': self.video_infos[idx].label}
        # return self.video_infos[idx]['ann']

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            # img_info = self.img_infos[i]
            # if img_info['width'] / img_info['height'] > 1:
            self.flag[i] = 1

    def _load_image(self, directory, image_tmpl, modality, idx):
        if modality in ['RGB', 'RGBDiff']:
            return [mmcv.imread(osp.join(directory, image_tmpl.format(idx)))]
        elif modality == 'Flow':
            x_imgs = mmcv.imread(
                osp.join(directory, image_tmpl.format('x', idx)),
                flag='grayscale')
            y_imgs = mmcv.imread(
                osp.join(directory, image_tmpl.format('y', idx)),
                flag='grayscale')
            return [x_imgs, y_imgs]
        else:
            raise ValueError(
                'Not implemented yet; modality should be '
                '["RGB", "RGBDiff", "Flow"]')

    def _sample_indices(self, record):
        '''

        :param record: VideoRawFramesRecord
        :return: list, list
        '''
        average_duration = (record.num_frames -
                            self.old_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)),
                                  average_duration)
            offsets = offsets + np.random.randint(average_duration,
                                                  size=self.num_segments)
        elif record.num_frames > max(self.num_segments, self.old_length):
            offsets = np.sort(np.random.randint(
                record.num_frames - self.old_length + 1,
                size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        if self.temporal_jitter:
            skip_offsets = np.random.randint(
                self.new_step, size=self.old_length // self.new_step)
        else:
            skip_offsets = np.zeros(
                self.old_length // self.new_step, dtype=int)
        return offsets + record.start_frame, skip_offsets  # frame index starts from 1

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.old_length - 1:
            tick = (record.num_frames - self.old_length + 1) / \
                float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x)
                                for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        if self.temporal_jitter:
            skip_offsets = np.random.randint(
                self.new_step, size=self.old_length // self.new_step)
        else:
            skip_offsets = np.zeros(
                self.old_length // self.new_step, dtype=int)
        return offsets + record.start_frame, skip_offsets

    def _get_test_indices(self, record):
        if record.num_frames > self.old_length - 1:
            tick = (record.num_frames - self.old_length + 1) / \
                float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x)
                                for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        if self.temporal_jitter:
            skip_offsets = np.random.randint(
                self.new_step, size=self.old_length // self.new_step)
        else:
            skip_offsets = np.zeros(
                self.old_length // self.new_step, dtype=int)
        return offsets + record.start_frame, skip_offsets

    def _get_frames(self, record, image_tmpl, modality, indices, skip_offsets):
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i, ind in enumerate(range(0, self.old_length, self.new_step)):
                if p + skip_offsets[i] <= record.end_frame:
                    seg_imgs = self._load_image(osp.join(
                        self.img_prefix, record.path),
                        image_tmpl, modality, p + skip_offsets[i])
                else:
                    seg_imgs = self._load_image(
                        osp.join(self.img_prefix, record.path),
                        image_tmpl, modality, p)
                images.extend(seg_imgs)
                if p + self.new_step < record.end_frame:
                    p += self.new_step
        return images

    def __getitem__(self, idx):
        record = self.video_infos[idx]
        if self.test_mode:
            segment_indices, skip_offsets = self._get_test_indices(record)
        else:
            segment_indices, skip_offsets = self._sample_indices(
                record) if self.random_shift else self._get_val_indices(record)

        data = dict(num_modalities=DC(to_tensor(len(self.modalities))),
                    gt_label=DC(to_tensor(record.label), stack=True,
                                pad_dims=None))

        verb_dist, noun_dist = self.visits_to_intdist(record.visits)
        if record.node==-1:
            vmask = (verb_dist>0).float()
            nmask = (noun_dist>0).float()
        else:
            vmask = torch.ones(verb_dist.shape)
            nmask = torch.ones(noun_dist.shape)

        data.update(dict(
                        verb_dist=DC(to_tensor(verb_dist), stack=True, pad_dims=None),
                        noun_dist=DC(to_tensor(noun_dist), stack=True, pad_dims=None),
                        # int_dist=DC(to_tensor(record.int_dist), stack=True, pad_dims=None),
                        vmask=DC(to_tensor(vmask), stack=True, pad_dims=None),
                        nmask=DC(to_tensor(nmask), stack=True, pad_dims=None),
                        )
                    )


        # handle the first modality
        modality = self.modalities[0]
        image_tmpl = self.image_tmpls[0]
        img_group = self._get_frames(
            record, image_tmpl, modality, segment_indices, skip_offsets)

        flip = True if np.random.rand() < self.flip_ratio else False
        if (self.img_scale_dict is not None
                and record.path in self.img_scale_dict):
            img_scale = self.img_scale_dict[record.path]
        else:
            img_scale = self.img_scale
        (img_group, img_shape, pad_shape,
         scale_factor, crop_quadruple) = self.img_group_transform(
            img_group, img_scale,
            crop_history=None,
            flip=flip, keep_ratio=self.resize_keep_ratio,
            div_255=self.div_255,
            is_flow=True if modality == 'Flow' else False)
        # ori_shape = (256, 340, 3)
        img_meta = dict(
            # ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            crop_quadruple=crop_quadruple,
            flip=flip)
        # [M x C x H x W]
        # M = 1 * N_oversample * N_seg * L
        if self.input_format == "NCTHW":
            img_group = img_group.reshape(
                (-1, self.num_segments, self.new_length) + img_group.shape[1:])
            # N_over x N_seg x L x C x H x W
            img_group = np.transpose(img_group, (0, 1, 3, 2, 4, 5))
            # N_over x N_seg x C x L x H x W
            img_group = img_group.reshape((-1,) + img_group.shape[2:])
            # M' x C x L x H x W

        data.update(dict(
            img_group_0=DC(to_tensor(img_group), stack=True, pad_dims=2),
            img_meta=DC(img_meta, cpu_only=True)
        ))

        # handle the rest modalities using the same
        for i, (modality, image_tmpl) in enumerate(
                zip(self.modalities[1:], self.image_tmpls[1:])):
            img_group = self._get_frames(
                record, image_tmpl, modality, segment_indices, skip_offsets)

            # apply transforms
            flip = True if np.random.rand() < self.flip_ratio else False
            (img_group, img_shape, pad_shape,
             scale_factor, crop_quadruple) = self.img_group_transform(
                img_group, img_scale,
                crop_history=data['img_meta'][
                    'crop_quadruple'],
                flip=data['img_meta'][
                    'flip'], keep_ratio=self.resize_keep_ratio,
                div_255=self.div_255,
                is_flow=True if modality == 'Flow' else False)

            if self.input_format == "NCTHW":
                # Convert [M x C x H x W] to [M' x C x T x H x W]
                # M = 1 * N_oversample * N_seg * L
                # M' = 1 * N_oversample * N_seg, T = L
                img_group = img_group.reshape(
                    (-1, self.num_segments,
                     self.new_length) + img_group.shape[1:])
                img_group = np.transpose(img_group, (0, 1, 3, 2, 4, 5))
                img_group = img_group.reshape((-1,) + img_group.shape[2:])

            data.update({
                'img_group_{}'.format(i+1):
                DC(to_tensor(img_group), stack=True, pad_dims=2),
            })

        return data
