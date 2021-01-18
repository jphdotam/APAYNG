from collections import defaultdict

import torch
import torch.nn as nn

from lib.data import LABELS
from lib.hrnet import get_config, get_seg_model


def load_model(cfg):
    hrnet_cfg = get_config(len(LABELS))
    model = get_seg_model(hrnet_cfg)
    device = cfg['training']['device']

    if modelpath := cfg['resume'].get('path', None):
        state = torch.load(modelpath)
        model.module.load_state_dict(state['state_dict'])
        starting_epoch = state['epoch']
        if conf_epoch := cfg['resume'].get('epoch', None):
            print(
                f"WARNING: Loaded model trained for {starting_epoch - 1} epochs but config explicitly overrides to {conf_epoch}")
            starting_epoch = conf_epoch
    else:
        starting_epoch = 1
        state = {}

    model = model.to(device)

    if cfg['training']['data_parallel']:
        model = nn.DataParallel(model)

    return model, starting_epoch, state
