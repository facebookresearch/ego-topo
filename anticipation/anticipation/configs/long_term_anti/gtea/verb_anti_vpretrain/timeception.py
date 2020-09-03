# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from common import *

model = dict(
    type='TimeXModel',
    num_classes=num_classes,
    backbone=dict(
        type="Timeception",
        input_shape=(256, 2048, 64, 1, 1),
    ),
    fc_indim=5000,
)

data = set_dataset_type(data, 'EpicFutureLabelsI3D')
work_dir = get_workdir(__file__)
