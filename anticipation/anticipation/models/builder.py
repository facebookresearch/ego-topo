# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from torch import nn

import mmcv
from mmaction.models.builder import build

from .registry import GFB_MODULES, LFB_MODULES, TIMECEPTION


def build_lfb_module(cfg):
    return build(cfg, LFB_MODULES)

def build_gfb_module(cfg):
    return build(cfg, GFB_MODULES)

def build_timeception(cfg):
    return build(cfg, TIMECEPTION)
