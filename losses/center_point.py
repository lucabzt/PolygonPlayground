"""
Mean Absolute Error of the two centroids of the polygons
"""

import torch
from torch import nn

from util import get_random_poly, sort_poly


class CentroidLoss(nn.Module):
    def __init__(self):
        super(CentroidLoss, self).__init__()

    @staticmethod
    def get_area(polys: torch.Tensor) -> torch.Tensor:
        """
        Finds the batched area of all the polygons given in polys
        Formula taken from here:
        https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
        """
        b, n, _ = polys.shape
        x_coords = polys[:, :, 0].reshape(b, n)
        y_coords = polys[:, :, 1].reshape(b, n)
        return 0.5*torch.abs((x_coords * torch.roll(y_coords, 1, dims=1)).sum(dim=1) -
                             (y_coords * torch.roll(x_coords, 1, dims=1)).sum(dim=1))

    @staticmethod
    def get_mean(poly: torch.Tensor) -> torch.Tensor:
        return torch.mean(poly, dim=1, keepdim=True)

    @staticmethod
    def find_centroid(poly: torch.Tensor) -> torch.Tensor:
        """
        Implements the centroid formula from wikipedia:
        https://en.wikipedia.org/wiki/Centroid#Of_a_polygon
        """
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
        means_a = self.get_mean(a)
        means_b = self.get_mean(b)
        areas_a = self.get_area(a)
        areas_b = self.get_area(b)
        centroid_loss = torch.abs(centroids_a - centroids_b).mean()
        mean_loss = torch.abs(means_a - means_b).mean()
        area_loss = torch.abs(areas_a - areas_b).mean()

        return 0.5 * (centroid_loss + mean_loss + area_loss)


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from shapely.geometry import Polygon
    torch.random.manual_seed(0)

    test = get_random_poly().unsqueeze(0) / 500
    test[0, 2], test[0, 3] = test[0, 3].clone(), test[0, 2].clone()
    test_sorted = sort_poly(test[0].clone()).unsqueeze(0)
    print(f"SELF CALCULATED AREA: {CentroidLoss.get_area(test)}")
    print(f"SELF CALCULATED AREA SORTED: {CentroidLoss.get_area(test_sorted)}")
    print(f"SHAPELY AREA: {Polygon(test[0]).area}")
    print(f"SHAPELY AREA SORTED: {Polygon(test_sorted[0]).area}")
    centroid = CentroidLoss.find_centroid(test)

    test = test.squeeze(0)  # Remove batch dimension if needed
    test_closed = torch.cat([test, test[:1]], dim=0)  # Ensure it forms a closed shape

    centroid_plot = centroid.squeeze() * 500

    plt.plot(test_closed[:, 0] * 500, test_closed[:, 1] * 500, color='blue')
    plt.plot(test_sorted[0, :, 0] * 500, test_sorted[0, :, 1] * 500, color='red')
    plt.scatter(centroid_plot[0], centroid_plot[1], color="red", marker="x")

    plt.xlim([0, 500])
    plt.ylim([0, 500])
    plt.axis('equal')

    plt.show()