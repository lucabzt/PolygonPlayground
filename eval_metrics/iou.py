"""
Implements non-differentiable iou for two polygons
"""
from shapely.geometry import Polygon
import torch
import torch.nn as nn

class IoU(nn.Module):
    """
    Implements non-differentiable iou for two polygons
    """
    @torch.no_grad()
    def forward(self, a, b) -> float:
        a_poly = Polygon(a)
        b_poly = Polygon(b)
        polygon_intersection = a_poly.intersection(b_poly).area
        polygon_union = a_poly.area + b_poly.area - polygon_intersection
        return float(polygon_intersection / polygon_union)