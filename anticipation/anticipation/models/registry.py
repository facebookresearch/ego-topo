# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch.nn as nn

from mmaction.models.registry import Registry

LFB_MODULES = Registry('lfb_modules')
GFB_MODULES = Registry('gfb_modules')
TIMECEPTION = Registry('timeception')
