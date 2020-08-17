# model settings
anticipation = True
anti2 = True

model = dict(
    type='NodeLSTM',
    backbone=dict(
        type='lstm',
        num_layers=1,
        in_channels=2048,
        out_channels=2048,
    ),
    # reg_head=dict(
    #     type='RegHead',
    #     in_channels=2048,
    #     out_channels=2048,)
)
train_cfg = None
test_cfg = None
# dataset settings
dataset_type = 'EpicNodeDatasetGFB'
mode="_verb" # _verb|_noun|''
split="S1"
# feature_file = "data/features/{}/".format(split) + "{}_lfb_s30.pkl"
feature_file = "data/features/{}/".format(split) + "{}_lfb.pkl"
fb_file = "data/features/{}/".format(split) + "{}_lfb.pkl"
graph_root = 'data/graphs/{}/'.format(split)
data = dict(
    videos_per_gpu=32,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        feature_file=feature_file.format("train"),
        ann_file='data/epic/split/train_{}{}.csv'.format(split, mode),
        fb=fb_file.format("train"),
        graph_root=graph_root,
        max_length=8,
        anti2=anti2,
        anticipation_task=anticipation),
    val=dict(
        type=dataset_type,
        feature_file=feature_file.format("val"),
        ann_file='data/epic/split/val_{}{}.csv'.format(split, mode),
        fb=fb_file.format("val"),
        graph_root=graph_root,
        max_length=8,
        anti2=anti2,
        anticipation_task=anticipation),
    test=dict(
        type=dataset_type,
        feature_file=feature_file.format("val"),
        ann_file='data/epic/split/val_{}{}.csv'.format(split, mode),
        fb=fb_file.format("val"),
        graph_root=graph_root,
        max_length=8,
        anti2=anti2,
        anticipation_task=anticipation))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    step=[15, 20])
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
total_epochs = 25
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'work_dir/lstm/verb_noun/nodelstm/'
load_from = None
resume_from = None
