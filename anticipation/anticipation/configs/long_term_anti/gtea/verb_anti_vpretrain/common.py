# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

dset = 'gtea'

label = 'verb'
num_classes = 19

task = 'anticipation'
val_timestamps = [0.25, 0.5, 0.75]
train_timestamps = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

train_many_shot = False
gfb_loss_weight = 0.0

train_cfg = None
test_cfg = None

# dataset settings
dataset_type = None
mode = "" # _verb|_noun|''
split = "S1"
pretrain = "verb"
feature_file = "data/{}/features/".format(dset) + "{{}}_lfb_s30_{}.pkl".format(pretrain)
fb_file = "data/{}/features/".format(dset) + "{{}}_lfb_s30_{}.pkl".format(pretrain)
graph_root = 'data/{}/graphs/'.format(dset)
node_num_member = 8
rand_visit = False
graph_aug = True
graph_drop = 0.5
gfb_loss_weight = 0.0

data = dict(
    videos_per_gpu=256,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file='data/{}/split/train_{}{}.csv'.format(dset, split, mode),
        label=label,
        task=task,
        dset=dset,
        fb=fb_file.format("train"),
        graph_root=graph_root,
        train_timestamps=train_timestamps,
        val_timestamps=val_timestamps,
        train_many_shot=train_many_shot,
        node_num_member=node_num_member,
        rand_visit=rand_visit,
        graph_aug=graph_aug,
        graph_drop=graph_drop,
        test_mode=False),
    val=dict(
        type=dataset_type,
        ann_file='data/{}/split/val_{}{}.csv'.format(dset, split, mode),
        label=label,
        task=task,
        dset=dset,
        fb=fb_file.format("val"),
        graph_root=graph_root,
        train_timestamps=train_timestamps,
        val_timestamps=val_timestamps,
        train_many_shot=train_many_shot,
        node_num_member=node_num_member,
        test_mode=True),
    test=dict(
        type=dataset_type,
        ann_file='data/{}/split/val_{}{}.csv'.format(dset, split, mode),
        label=label,
        task=task,
        dset=dset,
        fb=fb_file.format("val"),
        train_timestamps=train_timestamps,
        val_timestamps=val_timestamps,
        train_many_shot=train_many_shot,
        graph_root=graph_root,
        node_num_member=node_num_member,
        test_mode=True)
    )


#optimizer
optimizer = dict(type='Adam', lr=1e-3, weight_decay=1e-5)
optimizer_config = dict()

# learning policy
lr_config = dict(
    policy='step',
    step=[])
checkpoint_config = dict(interval=10)
workflow = [('train', 5), ('val', 1)]

# yapf:disable
log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

# runtime settings
total_epochs = 50
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None

def get_workdir(name):
    name = name[name.find("configs/") + 8:name.rfind(".py")]
    work_dir = 'work_dir/{}'.format(name)
    return work_dir

def set_dataset_type(data_dict, dataset_type):
    for split in ['train', 'val', 'test']:
        data_dict[split]['type'] = dataset_type
    return data_dict



