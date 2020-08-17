from torch import nn

import mmcv
from mmaction.models.builder import build

from .registry import GFB_MODULES, LFB_MODULES


def build_lfb_module(cfg):
    return build(cfg, LFB_MODULES)

def build_gfb_module(cfg):
    return build(cfg, GFB_MODULES)
