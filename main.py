"""
Main function for testing the experiment
"""
import torch

from losses.approx_hausdorff import AveragedHausdorffLoss
from losses.center_point import CentroidLoss
from losses.chamfer import ChamferDistance
from losses.hungarian_l1 import HungarianL1Loss
from losses.logger import MetricLogger
from losses.mask_l1 import MaskL1Loss
from util import *
from optimizer import Optimizer
from eval_metrics.iou import IoU
from eval_metrics.hiou import HIoU
from vis_utils import *
import time

torch.random.manual_seed(0)

def run_without_visuals(optim):
    optim.n_steps(optim.scheduler.num_epochs)
    optim.logger.plot()

def run(optim):
    stats = optim.logger.curr_stats
    stats['epoch'] = optim.scheduler.epoch
    f = get_plot(torch.concat([optim.preds, optim.targets], dim=0), stats)
    plt.show()

    for i in range(optim.scheduler.num_epochs):
        stats = optim.logger.curr_stats
        stats['epoch'] = optim.scheduler.epoch
        optim.step()
        update_plot(f, torch.concat([optim.preds, optim.targets], dim=0), stats)
        time.sleep(0.4)

    optim.logger.plot()

def main():
    vis = True
    epochs = 100
    loss_fn = AveragedHausdorffLoss()

    target = get_random_rect(center=[250, 250]).unsqueeze(0)
    pred = get_random_poly(center=[10, 10], r=(5, 5)).unsqueeze(0)

    optim = Optimizer(loss_fn, pred, target, num_epochs=epochs, lr=0.1, loss_steps = [0.3, 0.39])
    with MetricLogger(optim) as logger:
        optim.logger = logger
        if vis:
            run(optim)
        else:
            run_without_visuals(optim)



if __name__ == '__main__':
    main()
