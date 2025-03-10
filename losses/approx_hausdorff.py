"""
Approximate Hausdorff loss from
https://github.com/HaipengXiong/weighted-hausdorff-loss/blob/master/object-locator/losses.py
"""
from torch import nn
import torch
from torch import cdist

class AveragedHausdorffLoss(nn.Module):
    def __init__(self):
        super(AveragedHausdorffLoss, self).__init__()

    def get_hausdorff(self, set1, set2):
        """
        Compute the Averaged Hausdorff Distance function
         between two unordered sets of points (the function is symmetric).
         Batches are not supported, so squeeze your inputs first!
        :param set1: Tensor where each row is an N-dimensional point.
        :param set2: Tensor where each row is an N-dimensional point.
        :return: The Averaged Hausdorff Distance between set1 and set2.
        """
        set1 = set1.squeeze()
        set2 = set2.squeeze()

        assert set1.ndimension() == 2, 'got %s' % set1.ndimension()
        assert set2.ndimension() == 2, 'got %s' % set2.ndimension()

        assert set1.size()[1] == set2.size()[1], \
            'The points in both sets must have the same number of dimensions, got %s and %s.' \
            % (set2.size()[1], set2.size()[1])

        d2_matrix = cdist(set1.float(), set2.float())

        # Modified Chamfer Loss
        term_1 = torch.mean(torch.min(d2_matrix, 1)[0])
        term_2 = torch.mean(torch.min(d2_matrix, 0)[0])

        res = term_1 + term_2

        return res

    def forward(self, set1, set2):
        assert set1.dim() == 3, f"input should be batched, is {set1.shape}"
        batch_size = set1.shape[0]
        loss = 0.
        for idx in range(batch_size):
            loss += self.get_hausdorff(set1[idx], set2[idx])

        return loss / batch_size
