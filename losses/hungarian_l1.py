"""
This implements l1 loss with the help of hungarian point matching
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F


class HungarianL1Loss(nn.Module):
    def __init__(self):
        super(HungarianL1Loss, self).__init__()
        self.point_matcher = PointMatcher()

    def forward(self, pred, target):
        batch_size = pred.shape[0]
        loss = 0.
        for batch_idx in range(batch_size):
            pred_ind, _ = self.point_matcher(pred[batch_idx], target[batch_idx])
            pred_sorted = pred[batch_idx][pred_ind]
            l1_loss = F.l1_loss(pred_sorted, target[batch_idx])
            loss = loss + l1_loss

        return (loss/batch_size)  # TODO somehow normalize loss to 0 and 1

class PointMatcher(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, ground_truth):
        """
        Expects prediction and ground_truth to be in shape [n_polygons, n_points, 2]
        """
        # Compute the point-wise L1 loss to get the cost matrix
        cost_matrix = torch.cdist(ground_truth.float(), prediction.float(), p=1)

        row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
        return col_ind, cost_matrix[row_ind, col_ind].sum()