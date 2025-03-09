"""
This implements l1 loss with the help of hungarian point matching
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F

from losses.hungarian_l1 import HungarianL1Loss


class L1Metric(nn.Module):
    def __init__(self):
        super(L1Metric, self).__init__()
        self.l1_loss = HungarianL1Loss()

    @torch.no_grad()
    def forward(self, pred, target):
        return self.l1_loss(pred, target).item()