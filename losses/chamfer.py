"""
Implementation of the Chamfer distance between two points.
Copied from https://github.com/ThibaultGROUEIX/ChamferDistancePytorch/blob/master/chamfer_python.py
"""

import torch

class ChamferDistance(torch.nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device

    def forward(self, a, b):
        return self.chamferLoss(a, b)

    @staticmethod
    def pairwise_dist(x, y):
        xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
        rx = xx.diag().unsqueeze(0).expand_as(xx)
        ry = yy.diag().unsqueeze(0).expand_as(yy)
        P = rx.t() + ry - 2 * zz
        return P

    @staticmethod
    def NN_loss(x, y, dim=0):
        dist = ChamferDistance.pairwise_dist(x, y)
        values, indices = dist.min(dim=dim)
        return values.mean()

    @staticmethod
    def batched_pairwise_dist(a, b):
        x, y = a.double(), b.double()
        bs, num_points_x, points_dim = x.size()
        bs, num_points_y, points_dim = y.size()

        xx = torch.pow(x, 2).sum(2)
        yy = torch.pow(y, 2).sum(2)
        zz = torch.bmm(x, y.transpose(2, 1))
        rx = xx.unsqueeze(1).expand(bs, num_points_y, num_points_x) # Diagonal elements xx
        ry = yy.unsqueeze(1).expand(bs, num_points_x, num_points_y) # Diagonal elements yy
        P = rx.transpose(2, 1) + ry - 2 * zz
        return P

    @staticmethod
    def distChamfer(a, b):
        """
        :param a: Pointclouds Batch x nul_points x dim
        :param b:  Pointclouds Batch x nul_points x dim
        :return:
        -closest point on b of points from a
        -closest point on a of points from b
        -idx of closest point on b of points from a
        -idx of closest point on a of points from b
        Works for pointcloud of any dimension
        """
        P = ChamferDistance.batched_pairwise_dist(a, b)
        return torch.min(P, 2)[0].float(), torch.min(P, 1)[0].float(), torch.min(P, 2)[1].int(), torch.min(P, 1)[1].int()

    @staticmethod
    def chamferLoss(a, b):
        """
            :param a: Pointclouds Batch x nul_points x dim
            :param b:  Pointclouds Batch x nul_points x dim
            :return:
            - chamfer distance loss between a and b
        """
        cd1, cd2, _, _ = ChamferDistance.distChamfer(a, b)
        return cd1.mean() + cd2.mean()

    @staticmethod
    def chamfer_2d_fast(pc1, pc2):
        # pc1: (B, N, 2), pc2: (B, M, 2)
        dist = torch.cdist(pc1, pc2, p=2)  # L2 distance matrix: (B, N, M)
        loss = dist.min(dim=2)[0].mean() + dist.min(dim=1)[0].mean()
        return loss