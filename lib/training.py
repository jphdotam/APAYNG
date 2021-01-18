import os
from collections import deque
from contextlib import nullcontext

import wandb

import torch
import torch.nn.functional as F

from lib.data import LABELS
from lib.losses import dice_loss, jaccard_loss
from lib.hrnet import upsample_pred_if_needed
from lib.metrics import mIOU

class Am:
    """Simple average meter which stores progress as a running average"""

    def __init__(self, n_for_running_average=100):  # n is in samples not batches
        self.n_for_running_average = n_for_running_average
        self.val, self.avg, self.sum, self.count, self.running, self.running_average = None, None, None, None, None, None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.running = deque(maxlen=self.n_for_running_average)
        self.running_average = -1

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.running.extend([val] * n)
        self.count += n
        self.avg = self.sum / self.count
        self.running_average = sum(self.running) / len(self.running)


def cycle(train_or_test, model, dataloader, epoch, criterion, optimizer, cfg, scheduler):
    meter_loss, meter_iou_loss, meter_iou_score = Am(), Am(), Am()

    log_freq = cfg['output']['log_freq']
    device = cfg['training']['device']
    model = model.to(device)

    if train_or_test == 'train':
        model.train()
        training = True
    elif train_or_test == 'test':
        model.eval()
        training = False
    else:
        raise ValueError(f"train_or_test must be 'train', or 'test', not {train_or_test}")

    for i_batch, (x, y_true, _filename) in enumerate(dataloader):
        # Forward pass
        optimizer.zero_grad()

        x = x.to(device, non_blocking=True)
        y_true = y_true.to(device, non_blocking=True)

        # Forward pass
        with nullcontext() if training else torch.no_grad():
            y_pred = model(x)
            y_pred = upsample_pred_if_needed(y_pred, y_true)
            loss = criterion(y_pred, y_true)

        # Backward pass
        if training:
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

        # Metrics
        with torch.no_grad():
            iou_score = mIOU(y_true, y_pred, num_classes=len(LABELS))
            iou_loss = jaccard_loss(y_true, y_pred).cpu().numpy()

        meter_loss.update(loss, x.size(0))
        meter_iou_score.update(iou_score, x.size(0))
        meter_iou_loss.update(iou_loss, x.size(0))

        # Loss intra-epoch printing
        if (i_batch + 1) % log_freq == 0:

            print(f"{train_or_test.upper(): >5} [{i_batch + 1:03d}/{len(dataloader):03d}]\t\t"
                  f"IoU: {meter_iou_score.running_average:.5f}\t\t"
                  f"Loss: {meter_loss.running_average:.5f}\t\t"
                  f"IoU loss: {meter_iou_loss.running_average:.5f}\t\t")

            if training:
                wandb.log({"batch": len(dataloader) * epoch + i_batch,
                           f"iou_{train_or_test}": meter_iou_score.running_average,
                           f"loss_{train_or_test}": meter_loss.running_average,
                           f"iouloss_{train_or_test}": meter_iou_loss.running_average})

    print(f"{train_or_test.upper(): >5} Complete!\t\t"
          f"IoU: {meter_iou_score.avg:.5f}\t\t"
          f"Loss: {meter_loss.avg:.5f}\t\t"
          f"IoU loss: {meter_iou_loss.avg:.5f}")

    loss = float(meter_loss.avg.detach().cpu().numpy())

    wandb.log({"epoch": epoch,
               f"iou_{train_or_test}": meter_iou_score.avg,
               f"loss_{train_or_test}": meter_loss.avg,
               f"iouloss_{train_or_test}": meter_iou_loss.avg})

    return loss


def save_state(state, save_name, test_metric, best_metric, cfg, last_save_path, lowest_best=True, force=False):
    save = cfg['output']['save']
    save_dir = cfg['paths']['models']
    save_path = os.path.join(save_dir, save_name)

    if save == 'all' or force:
        torch.save(state, save_path)
    elif (test_metric < best_metric) == lowest_best:
        print(f"{test_metric:.5f} better than {best_metric:.5f} -> SAVING")
        if save == 'best':  # Delete previous best if using best only; otherwise keep previous best
            if last_save_path:
                try:
                    os.remove(last_save_path)
                except FileNotFoundError:
                    print(f"Failed to find {last_save_path}")
        best_metric = test_metric
        torch.save(state, save_path)
        last_save_path = save_path
    else:
        print(f"{test_metric:.5g} not improved from {best_metric:.5f}")
    return best_metric, last_save_path




