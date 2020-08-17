import os

gfb_pool = "next"
anticipation = True
gfb_next = True
anti2 = True
node_num_member = 10
# model settings
model = dict(
    type='Recognizer',
    backbone=dict(
        type='mlp',
        num_layers=1,
        in_channels=2048 + 2048,
        h_channels=2048,
        out_channels=2048,
    ),
    gfb_module=dict(
        type='SimpleGfbModule',
        pool_type=gfb_pool,
    ),
    cls_head=dict(
        type='MultiClsHead',
        with_avg_pool=False,
        temporal_feature_size=1,
        spatial_feature_size=1,
        dropout_ratio=0,
        in_channels=2048,
        num_classes=[125]))
train_cfg = None
test_cfg = None
# dataset settings
dataset_type = 'EpicFeatureDatasetGFB'
mode="_verb" # _verb|_noun|''
split="S1"
feature_file = "data/features/{}/".format(split) + "{}_lfb_s30.pkl"
fb_file = "data/features/{}/".format(split) + "{}_lfb_s30.pkl"
graph_root = 'data/graphs/{}/'.format(split)
data = dict(
    videos_per_gpu=256,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file='data/epic/split/train_{}{}.csv'.format(split, mode),
        feature_file=feature_file.format("train"),
        fb=fb_file.format("train"),
        graph_root=graph_root,
        test_mode=False,
        node_num_member=node_num_member,
        add_next_node=gfb_next,
        anti2=anti2,
        anticipation_task=anticipation),
    val=dict(
        type=dataset_type,
        ann_file='data/epic/split/val_{}{}.csv'.format(split, mode),
        feature_file=feature_file.format("val"),
        fb=fb_file.format("val"),
        graph_root=graph_root,
        test_mode=True,
        node_num_member=node_num_member,
        add_next_node=gfb_next,
        anti2=anti2,
        anticipation_task=anticipation),
    test=dict(
        type=dataset_type,
        ann_file='data/epic/split/val_{}{}.csv'.format(split, mode),
        feature_file=feature_file.format("val"),
        fb=fb_file.format("val"),
        graph_root=graph_root,
        test_mode=True,
        node_num_member=node_num_member,
        add_next_node=gfb_next,
        anti2=anti2,
        anticipation_task=anticipation))
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    step=[30, 40])
checkpoint_config = dict(interval=1)
# workflow = [('train', 5), ('val', 1)]
workflow = [('train', 1), ('val', 1)]
# yapf:disable
log_config = dict(
    interval=20,
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
work_dir = 'work_dir/{}'.format(name)
load_from = None
resume_from = None
