"""
This calculates iou first but falls back on hausdorff distance if iou is zero
"""
import math

import torch
from torch import nn

from eval_metrics.hausdorff import Hausdorff
from eval_metrics.iou import IoU

class HIoU(nn.Module):
    """
    This is a combination of IoU and Hausdorff.
    The formula is as follows:
    nh = normalized hausdorff distance
    L = (iou + (1-nh)) / 2
    """
    def __init__(self):
        super(HIoU, self).__init__()
        self.iou = IoU()
        self.hausdorff = Hausdorff()

    @torch.no_grad()
    def forward(self, a, b):
        """
        Expects a and b to be normalized and batched.
        """
        iou = self.iou(a,b)
        hd = self.hausdorff(a,b)
        hd_normalized = hd / 1
        return (1.5*iou + 0.5*(1-hd_normalized)) / 2
