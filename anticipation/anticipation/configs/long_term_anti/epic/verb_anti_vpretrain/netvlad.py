# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from common import *

centroids_path = "work_dir/long_term_anti/{}/data/{}_anti/centers.pkl".format(dset, label)
model = dict(
    type='NetVladModel',
    num_classes=num_classes,
    backbone=dict(
        type="NetVLAD",
        centroids_path=centroids_path,
        dim=2048,
        out_dim=2048,
        num_clusters=64,
        normalize_input=False,
        vladv2=True,
    ),
    fc_indim=2048,
)

data = set_dataset_type(data, 'EpicFutureLabelsI3D')
work_dir = get_workdir(__file__)

