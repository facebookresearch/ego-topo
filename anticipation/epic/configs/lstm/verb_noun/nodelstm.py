
# model settings
model = dict(
    type='NodeLSTM',
    backbone=dict(
        type='lstm',
        num_layers=1,
        in_channels=2048,
        out_channels=2048,
    ),
    reg_head=dict(
        type='RegHead',
        in_channels=2048,
        out_channels=2048,),
    cls_head=dict(
        type='MultiClsHead1D',
        in_channels=2048,
        num_classes=[125, 352])
)
train_cfg = None
test_cfg = None
# dataset settings
dataset_type = 'EpicNodeDataset'
# mode="_verb" # _verb|_noun|''
split="S1"
# feature_file = "data/features/{}/".format(split) + "{}_lfb_s30.pkl"
fb_file = "data/features/{}/".format(split) + "{}_lfb.pkl"
graph_root = 'data/graphs/{}/'.format(split)
data = dict(
    videos_per_gpu=256,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        fb=fb_file.format("train"),
        ann_file='data/epic/split/{}_{}.csv'.format("train", split),
        add_verb=True,
        add_noun=True,
        graph_root=graph_root,
        max_length=10),
    val=dict(
        type=dataset_type,
        fb=fb_file.format("val"),
        ann_file='data/epic/split/{}_{}.csv'.format("val", split),
        add_verb=True,
        add_noun=True,
        graph_root=graph_root,
        max_length=10),
    test=dict(
        type=dataset_type,
        fb=fb_file.format("val"),
        ann_file='data/epic/split/{}_{}.csv'.format("val", split),
        add_verb=True,
        add_noun=True,
        graph_root=graph_root,
        max_length=10))
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
name = __file__
name = name[name.find("configs/") + 8:name.rfind(".py")]
work_dir = 'work_dir/{}'.format(name)
load_from = None
resume_from = None
