# model settings
model = dict(
    type='Recognizer',
    backbone=dict(
        type='mlp',
        num_layers=2,
        in_channels=2048,
        h_channels=2048,
        out_channels=2048,
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
dataset_type = 'EpicFeatureDataset'
mode="_verb" # _verb|_noun|''
split="S1"
feature_file = "data/features/{}/".format(split) + "{}_lfb.pkl"
data = dict(
    videos_per_gpu=32,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file='data/epic/split/train_{}{}.csv'.format(split, mode),
        feature_file=feature_file.format("train"),
        test_mode=False,
        anticipation_task=True),
    val=dict(
        type=dataset_type,
        ann_file='data/epic/split/val_{}{}.csv'.format(split, mode),
        feature_file=feature_file.format("val"),
        test_mode=True,
        anticipation_task=True),
    test=dict(
        type=dataset_type,
        ann_file='data/epic/split/val_{}{}.csv'.format(split, mode),
        feature_file=feature_file.format("val"),
        test_mode=True,
        anticipation_task=True))
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
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
work_dir = 'work_dir/fixfeature/fixfeatures_{}_2mlp'.format(split)
load_from = None
resume_from = None
