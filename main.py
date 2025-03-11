"""
Main function for testing the experiment
"""
from eval_metrics.ciou import CIoU
from losses.approx_hausdorff import AveragedHausdorffLoss
from logger import MetricLogger
from losses.center_point import CentroidLoss
from losses.chamfer import ChamferDistance
from losses.hungarian_l1 import HungarianL1Loss
from optimal_steps import l1_lrs, centroid_lrs, hausdorff_lrs, chamfer_lrs, hausdorff_01, l1_02
from util import *
from optimizer import Optimizer
from vis_utils import *
import time

torch.random.manual_seed(0)


def run_without_visuals(optim):
    plt.ioff()
    optim.n_steps(optim.scheduler.num_epochs)
    optim.logger.plot()


def run(optim, vis_ch):
    stats = optim.logger.curr_stats
    stats['epoch'] = optim.scheduler.epoch
    # this only visualizes the convex hull of the first batch
    points = torch.concat([optim.preds[0], optim.targets[0]], dim=0)
    convex_hull = CIoU.get_convex_hull(points)
    if vis_ch:
        f = get_plot([optim.preds[0], optim.targets[0], convex_hull], stats, p_styles=DEFAULT_POLYS, c_styles=DEFAULT_EDGES)
    else:
        f = get_plot([optim.preds[0], optim.targets[0]], stats)
    plt.show()

    for i in range(optim.scheduler.num_epochs):
        pred = optim.preds[0]
        target = optim.targets[0]
        stats = optim.logger.curr_stats
        stats['epoch'] = optim.scheduler.epoch
        optim.step()
        points = torch.concat([pred, target], dim=0)
        convex_hull = CIoU.get_convex_hull(points)
        if vis_ch:
            update_plot(f, [pred, target, convex_hull], stats, DEFAULT_POLYS, DEFAULT_EDGES)
        else:
            update_plot(f, [pred, target], stats)
        time.sleep(0.4)

    optim.logger.plot()


def main():
    vis = True
    vis_ch = True
    epochs = 100
    loss_fn = HungarianL1Loss()

    target = get_random_rect(center=[250, 250]).unsqueeze(0)
    pred = get_random_poly(center=[20, 20], r=(20, 20)).unsqueeze(0)

    optim = Optimizer(loss_fn, pred, target, num_epochs=epochs, lr=0.2, loss_steps=l1_02)
    with MetricLogger(optim) as logger:
        optim.logger = logger
        if vis:
            run(optim, vis_ch)
        else:
            run_without_visuals(optim)



if __name__ == '__main__':
    main()
