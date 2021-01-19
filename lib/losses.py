import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from itertools import filterfalse


def jaccard_loss(true, logits, eps=1e-7):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        jacc_loss: the Jaccard loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()

    return 1 - jacc_loss


# From Ian Pann
class DiceLoss(nn.Module):
    def __init__(self,
                 num_classes,
                 epsilon: float = 1e-12,
                 per_image: bool = True,
                 ignore_empty: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.per_image = per_image
        self.ignore_empty = ignore_empty

    def forward(self, p, t):
        N, C, H, W = p.shape
        p = torch.sigmoid(p)
        p = p.reshape(N * C, -1)

        # 1-HOT
        t = torch.eye(self.num_classes)[t.squeeze(1)]
        t = t.permute(0, 3, 1, 2).float()

        t = t.reshape(N * C, -1)
        if self.ignore_empty:
            mask = t.sum(-1)
            if (mask > 0).sum().item() == 0:
                return 0.5
            p = p[mask > 0]
            t = t[mask > 0]
        if self.per_image:
            loss = 1 - (2 * (p * t).sum(dim=-1) + self.epsilon) / (
                        (t ** 2).sum(dim=-1) + (p ** 2).sum(dim=-1) + self.epsilon)
            loss = loss.mean()
        else:
            loss = 1 - (2 * (p * t).sum() + self.epsilon) / ((t ** 2).sum() + (p ** 2).sum() + self.epsilon)
        return loss


class DiceCEHybridLoss(nn.Module):
    def __init__(self, num_classes, bce_weight: float, dice: str = 'v1', **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.bce_weight = bce_weight
        self.dsc_loss = DiceLoss(num_classes=num_classes)
        self.bce_loss = nn.CrossEntropyLoss()

    def forward(self, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.dsc_loss(p, t) + self.bce_weight * self.bce_loss(p, t)


# https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/LovaszSoftmax/lovasz_loss.py
def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class LovaszSoftmax(nn.Module):
    def __init__(self, reduction='mean'):
        super(LovaszSoftmax, self).__init__()
        self.reduction = reduction

    def prob_flatten(self, input, target):
        assert input.dim() in [4, 5]
        num_class = input.size(1)
        if input.dim() == 4:
            input = input.permute(0, 2, 3, 1).contiguous()
            input_flatten = input.view(-1, num_class)
        elif input.dim() == 5:
            input = input.permute(0, 2, 3, 4, 1).contiguous()
            input_flatten = input.view(-1, num_class)
        else:
            raise ValueError(f"input is of {input.dim()} dimensions - not supported")
        target_flatten = target.view(-1)
        return input_flatten, target_flatten

    def lovasz_softmax_flat(self, inputs, targets):
        num_classes = inputs.size(1)
        losses = []
        for c in range(num_classes):
            target_c = (targets == c).float()
            if num_classes == 1:
                input_c = inputs[:, 0]
            else:
                input_c = inputs[:, c]
            loss_c = (torch.autograd.Variable(target_c) - input_c).abs()
            loss_c_sorted, loss_index = torch.sort(loss_c, 0, descending=True)
            target_c_sorted = target_c[loss_index]
            losses.append(torch.dot(loss_c_sorted, torch.autograd.Variable(lovasz_grad(target_c_sorted))))
        losses = torch.stack(losses)

        if self.reduction == 'none':
            loss = losses
        elif self.reduction == 'sum':
            loss = losses.sum()
        else:
            loss = losses.mean()
        return loss

    def forward(self, inputs, targets):
        inputs, targets = self.prob_flatten(inputs, targets)
        losses = self.lovasz_softmax_flat(inputs, targets)
        return losses