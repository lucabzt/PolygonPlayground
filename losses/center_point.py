"""
Mean Absolute Error of the two centroids of the polygons
"""

import torch
from torch import nn

class CentroidLoss(nn.Module):
    def __init__(self):
        super(CentroidLoss, self).__init__()

    @staticmethod
    def get_area(poly: torch.Tensor) -> torch.Tensor:
        pass

    @staticmethod
    def find_centroid(poly: torch.Tensor) -> torch.Tensor:
        #TODO check correctness as might be order-dependent
        # Shift the vertices to get the next vertex for each edge, wrapping around
        next_poly = torch.roll(poly, shifts=-1, dims=1)

        # Calculate the cross term (x_i * y_{i+1} - x_{i+1} * y_i) for each edge
        cross = (poly[..., 0] * next_poly[..., 1] - next_poly[..., 0] * poly[..., 1])

        # Compute the signed area of each polygon
        area = 0.5 * cross.sum(dim=1)  # Shape [n_batches]

        # Calculate the terms needed for centroid components
        sum_cx = ((poly[..., 0] + next_poly[..., 0]) * cross).sum(dim=1)
        sum_cy = ((poly[..., 1] + next_poly[..., 1]) * cross).sum(dim=1)

        # Compute the centroid coordinates, handling division by area
        centroid_x = sum_cx / (6 * area)
        centroid_y = sum_cy / (6 * area)

        # Combine into the final tensor of shape [n_batches, 2]
        centroid = torch.stack([centroid_x, centroid_y], dim=1)

        return centroid

    def forward(self, a, b):
        centroids_a = self.find_centroid(a)
        centroids_b = self.find_centroid(b)

        return torch.abs(centroids_a - centroids_b).mean()