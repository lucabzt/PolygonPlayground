import torch
from torch import nn

from losses.utils.diff_ras import DifferentiableRasterization
from losses.utils.torch_contour import Contour_to_mask, Contour_to_distance_map


class MaskL1Loss(nn.Module):
    def __init__(self):
        super(MaskL1Loss, self).__init__()
        self.raster = DifferentiableRasterization(500, 'cpu', tau=2)
        self.dice = DiceLoss()
        self.l1 = nn.L1Loss()
    def forward(self, input, target):
        input_r = self.raster(input.unsqueeze(1) / 500)

        target_r = self.raster(target.unsqueeze(1) / 500)
        #import matplotlib.pyplot as plt
        #plt.imshow(target_r[0][0])
        #plt.show()
        return self.dice(input_r, target_r) * 1000



class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the Dice Loss for a batch of predictions and targets.

        Parameters:
        -----------
        pred : torch.Tensor
            The predicted probabilities or binary values of shape (batch_size, width, height).
        target : torch.Tensor
            The ground truth binary mask of shape (batch_size, width, height).

        Returns:
        --------
        torch.Tensor
            A scalar tensor representing the average Dice loss across the batch.
        """
        # Ensure predictions and targets are floating point
        pred = pred.float()
        target = target.float()

        # Flatten each image into a single long vector
        # Shape after flattening: (batch_size, width*height)
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        # Compute intersection
        intersection = (pred_flat * target_flat).sum(dim=1)  # sum over spatial dimensions per batch

        # Compute Dice coefficient per batch element
        dice_coeff = (2. * intersection + self.smooth) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + self.smooth)

        # Compute Dice loss (1 - Dice coefficient)
        dice_loss = 1 - dice_coeff

        # Return the average Dice loss over the batch
        return dice_loss.mean()