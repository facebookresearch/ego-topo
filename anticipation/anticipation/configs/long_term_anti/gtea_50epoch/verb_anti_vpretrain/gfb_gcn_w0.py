import os
import numpy as np

label = 'verb'
# label = 'int'
task = 'anticipation'
# task = 'anticipation'
dset = 'gtea'
# dset = 'gtea'
train_many_shot = False
gfb_loss_weight = 0.0

if dset=='gtea':
    if label=='noun':
        num_classes = 53
    elif label=='int':
        num_classes = 106
    elif label == 'verb':
        num_classes = 19
elif dset=='epic':
    if label=='noun':
        num_classes = 352
    elif label=='int':
        num_classes = 250
    elif label == 'verb':
        num_classes = 125

if task == 'recognition':
    val_timestamps = [0.25, 0.5, 0.75, 1.0]
    train_timestamps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
elif task == 'anticipation':
    val_timestamps = [0.25, 0.5, 0.75]
    train_timestamps = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# model settings
model = dict(
    type='GFBModelGCN1',
    num_classes=num_classes,
    gfb_loss_weight=gfb_loss_weight,
    pre_trans=dict(
        type="mlp1D",
        num_layers=1,
        in_channels=2048,
        h_channels=2048,
        out_channels=2048
    ),
    gfb_module=dict(
        type="GCN",
        in_dim=2048,
        h_dim=2048,
        num_layers=1,
        dropout=0.7,
    ),
    backbone=None,
)

train_cfg = None
test_cfg = None

# dataset settings
dataset_type = 'EpicFutureLabelsGFBAug'
mode="" # _verb|_noun|''
split="S1"
pretrain="verb"
feature_file = "data/{}/features/tushar_features_11_11/".format(dset) + "{{}}_lfb_s30_{}.pkl".format(pretrain)
fb_file = "data/{}/features/tushar_features_11_11/".format(dset) + "{{}}_lfb_s30_{}.pkl".format(pretrain)
graph_root = 'data/{}/graphs/{}/'.format(dset, split)
node_num_member=8
rand_visit=False
graph_aug=True
graph_drop=0.5
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

# optimizer
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
optimizer = dict(type='Adam', lr=1e-3, weight_decay=1e-5)
optimizer_config = dict()

# learning policy
lr_config = dict(
    policy='step',
    step=[])
checkpoint_config = dict(interval=10)
workflow = [('train', 5), ('val', 1)]
# workflow = [('train', 1), ('val', 1)]
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
name = __file__
name = name[name.find("configs/") + 8:name.rfind(".py")]
#name = 'future_label/gfb/%s'%np.random.randint(100)
work_dir = 'work_dir/{}'.format(name)
load_from = None
resume_from = None
