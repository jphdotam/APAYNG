from efficientunet import get_efficientunet_b0, Conv2dSamePadding

import torch
import torch.nn as nn

from lib.data import LABELS
from lib.hrnet import get_config, get_seg_model
from lib.unext import UneXt50


def load_model(cfg):
    model_type = cfg['training']['model']
    if model_type == 'hrnet':
        hrnet_cfg = get_config(len(LABELS))
        model = get_seg_model(hrnet_cfg)
    elif model_type == 'unext':
        model = UneXt50(n_inputs=1, n_outputs=len(LABELS))
    else:
        raise ValueError(f"Unknown model type: {model_type}")

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


if __name__ == "__main__":
    model = get_efficientunet_b0(out_channels=10)
