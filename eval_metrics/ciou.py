"""
Implement giou for polygons by using convex hull algorithm
"""

from scipy.spatial import ConvexHull
from torch import nn
from eval_metrics.iou import IoU


class CIoU(nn.Module):
    def __init__(self):
        super(CIoU, self).__init__()
        self.iou = IoU()

    def forward(self, x, y):
        pass