from collections import OrderedDict

import wandb

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lib.data import LABELS
from lib.hrnet import upsample_pred_if_needed

LABELS_DICT = {i: label for i, label in enumerate(LABELS)}


def vis(model, dataset, cfg, epoch):
    device = cfg['training']['device']
    n_vis = cfg['output']['n_vis']

    dataloader = DataLoader(dataset, n_vis, shuffle=False, num_workers=1, pin_memory=True)
    batch_x, batch_y_true, batch_filepaths = next(iter(dataloader))

    wandb_images = []

    with torch.no_grad():

        batch_y_pred = model(batch_x.to(device))

        # If using HRNet, we need to upsample our predictions
        batch_y_pred = upsample_pred_if_needed(batch_y_pred, batch_y_true)

        for i, (x, y_true, y_pred, filepath) in enumerate(zip(batch_x, batch_y_true, batch_y_pred, batch_filepaths)):
            # May need to add upsampling code

            y_pred_class = torch.argmax(y_pred, dim=0)

            wandb_image = wandb.Image(x.cpu().numpy()[0],  # 0th colour channel as grayscale
                                      masks={
                                          "predictions": {
                                              "mask_data": y_pred_class.cpu().numpy(),
                                              "class_labels": LABELS_DICT
                                          },
                                          "ground_truth": {
                                              "mask_data": y_true.cpu().numpy(),
                                              "class_labels": LABELS_DICT
                                          }
                                      })

            wandb_images.append(wandb_image)

        wandb.log({"epoch": epoch,
                   "images": wandb_images})
