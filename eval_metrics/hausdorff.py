"""
Implements the hausdorff distance metric
"""
import torch
from torch import nn
from scipy.spatial.distance import directed_hausdorff

class Hausdorff(nn.Module):
    def __init__(self):
        super(Hausdorff, self).__init__()

    @torch.no_grad()
    def forward(self, pred, target):
        score = 0
        for batch_idx in range(pred.shape[0]):
            a = pred[batch_idx]
            b = target[batch_idx]
            score += max(directed_hausdorff(a, b)[0],
                        directed_hausdorff(b, a)[0])
        return score / pred.shape[0]