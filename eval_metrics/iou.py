"""
Implements non-differentiable iou for two polygons
"""
from shapely.geometry import Polygon
import torch
import torch.nn as nn

from util import sort_poly


class IoU(nn.Module):
    """
    Implements non-differentiable iou for two polygons
    """
    def __init__(self):
        super(IoU, self).__init__()

    @torch.no_grad()
    def forward(self, pred, target) -> float:
        """
        mean iou over the whole batch
        """
        full_iou = 0.
        for batch_idx in range(pred.shape[0]):
            a = pred[batch_idx]
            b = target[batch_idx]
            a = sort_poly(a)
            b = sort_poly(b)
            a_poly = Polygon(a)
            b_poly = Polygon(b)
            polygon_intersection = a_poly.intersection(b_poly).area
            polygon_union = a_poly.union(b_poly).area
            full_iou += float(polygon_intersection / polygon_union)

        return full_iou / pred.shape[0]