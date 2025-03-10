"""
Implement giou for polygons by using convex hull algorithm
"""

import torch
from shapely import Polygon
from torch import nn
from scipy.spatial import ConvexHull
from eval_metrics.iou import IoU
from util import sort_poly


class CIoU(nn.Module):
    def __init__(self):
        super(CIoU, self).__init__()
        self.iou = IoU()

    @staticmethod
    def get_convex_hull(points):
        """
        Compute the convex hull of a set of points using scipy.spatial.ConvexHull.
        Returns a tensor/array of points forming the convex hull.
        """
        ch = ConvexHull(points)
        return points[ch.vertices]

    def get_ciou(self, a, b):
        """
        a and b are two polygons with shape [n_vertices, 2]
        """
        point_set = torch.concat([a, b], dim=0)
        assert point_set.dim() == 2
        convex_hull = self.get_convex_hull(point_set)
        ch_shapely = Polygon(convex_hull)
        a_shapely = Polygon(sort_poly(a))
        b_shapely = Polygon(sort_poly(b))
        ab_union = a_shapely.union(b_shapely).area
        iou = self.iou(a.unsqueeze(0), b.unsqueeze(0))
        ciou = iou - (ch_shapely.area - ab_union) / ch_shapely.area
        return ciou

    def forward(self, a, b):
        batch_size = a.shape[0]
        assert a.dim() == b.dim() == 3, f"dimensions dont match. {a.dim()} != {b.dim()}"
        ciou = 0.

        # sum ciou over all batches
        for idx in range(batch_size):
            ciou += self.get_ciou(a[idx], b[idx])

        return ciou / batch_size
