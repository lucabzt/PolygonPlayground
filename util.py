"""
Useful helper functions
"""
import copy
import csv

import torch
import numpy as np
import matplotlib.pyplot as plt

from poly_configs import *
import math

def sort_poly(poly) -> torch.Tensor:
    """
    :param poly: list/numpy array/tensor
    :return: torch polygon sorted by angle to center coordinate
    """
    poly_np = np.array(poly)
    centroid = np.array([poly_np[:, 0].sum(), poly_np[:, 1].sum()]) / len(poly_np)
    sorted_points = sorted(poly_np, key=lambda p: np.arctan2((p[1] - centroid[1]), p[0] - centroid[0]))

    return torch.tensor(np.array(sorted_points))

def loss_to_metric(loss):
    """
    Takes a loss function as input and returns the same loss but with a modified forward function,
    such that it returns a float instead of a tensor (can be useful for logging)
    """
    loss_fn = copy.deepcopy(loss)
    old_forward = loss_fn.forward
    def new_forward(a,b):
        return old_forward(a,b).item()
    loss_fn.forward = new_forward
    return loss_fn


def csv_to_dict_of_lists(csv_file_path):
    """
    Convert csv file to a dict of keys (first line values) and values (list of values)
    """
    with open(csv_file_path, "r", newline="") as f:
        reader = csv.reader(f)

        # Read the header (first line)
        header = next(reader)

        # Initialize a dictionary with empty lists
        data = {column_name: [] for column_name in header}

        # Fill the lists for each column
        for row in reader:
            for column_name, cell_value in zip(header, row):
                data[column_name].append(cell_value)

    return data

def rand(size, r1, r2) -> torch.Tensor:
    """
    Create random tensor in specified range
    """
    if r2 < r1:
        r1, r2 = r2, r1
    return (r2-r1) * torch.rand(size) + r1

def get_random_poly(n=4, size=500, center=None, r=(20, 40)) -> torch.Tensor:
    """
    :param n: amount of vertices
    :param size: needs to be in the specified size
    :param center: center of the range where the polygon can be
    :param r: range around the center where coordinates can be
    :return: randomly generated polygon
    """
    if center is None:
        center = rand(2, 0, size)
    center = torch.tensor(center, dtype=torch.float32) if not isinstance(center, torch.Tensor) else center
    # keep r in bounds
    r = [min(r[i], center[i].item(), size - center[i].item()) for i in range(2)]

    poly = []
    for vn in range(n):
        x = rand(1, center[0] - r[0], center[0] + r[0])
        y = rand(1, center[1] - r[1], center[1] + r[1])
        poly.append([x.item(), y.item()])

    return sort_poly(poly)

def get_random_rect(center=None, size=500, aspect_ratio=3, r=30) -> torch.Tensor:
    """
    Create a random rectangle with given aspect ratio, possibly rotated.
    :param center: Center of the rectangle (x, y)
    :param size: size of the plane where the rectangle lives
    :param aspect_ratio: Width / Height of the rectangle
    :param r: furthest distance from center to one of the points
    :return: 4x2 tensor of vertices sorted in clockwise order
    """
    # Edge cases
    if center is None:
        center = rand(2, 0, size)
    center_tensor = torch.tensor(center, dtype=torch.float32) if not isinstance(center, torch.Tensor) else center
    # keep r in bounds
    r = min([min(r, center_tensor[i].item(), size - center_tensor[i].item()) for i in range(2)])

    ar = aspect_ratio
    denominator = math.sqrt(ar ** 2 + 1)
    h = (2 * r) / denominator
    w = ar * h

    # Generate corners centered at (0,0)
    corners = torch.tensor([
        [-w/2, -h/2],
        [w/2, -h/2],
        [w/2, h/2],
        [-w/2, h/2]
    ], dtype=torch.float32)

    # Random rotation angle
    theta = torch.rand(1) * 2 * math.pi
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    # Rotation matrix
    rotation_matrix = torch.stack([
        torch.cat([cos_theta, -sin_theta]),
        torch.cat([sin_theta, cos_theta])
    ], dim=0)

    # Apply rotation
    rotated_corners = torch.mm(corners, rotation_matrix)

    # Translate to center
    rotated_corners += center_tensor

    # Sort the polygon vertices
    return sort_poly(rotated_corners)

if __name__ == '__main__':
    from vis_utils import vis_poly, get_plot
    from eval_metrics.iou import IoU
    from eval_metrics.hiou import HIoU

    target = get_random_rect(center=[250,250])
    pred = get_random_poly(center=[200,200], r=(30,30))
    metric = HIoU()
    iou = metric(target, pred)

    plt.close()
    get_plot([target, pred], stats={"iou": iou}, p_styles=[DEFAULT(), RED()], c_styles=[CIRCLE_CORNERS(s=50), CIRCLE_CORNERS(s=30, color="red")])
    plt.show()

