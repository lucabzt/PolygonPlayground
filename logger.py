"""
Class for logging different loss functions in csv data
"""
import csv
import os

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from eval_metrics.ciou import CIoU
from eval_metrics.iou import IoU
from eval_metrics.hiou import HIoU
from eval_metrics.hausdorff import Hausdorff
from eval_metrics.l1 import L1Metric
from util import loss_to_metric, csv_to_dict_of_lists


class MetricLogger:
    def __init__(self, optim):
        self.optim = optim
        self.eval_metrics = [loss_to_metric(optim.loss_fn)] + [
            CIoU(),
            IoU(),
            HIoU(),
            Hausdorff(),
            L1Metric(),
        ]
        self.loss_fn = optim.loss_fn
        self.name = type(optim.loss_fn).__name__.lower()
        os.makedirs(f'./results/{self.name}', exist_ok=True)
        self.file_name = f'./results/{self.name}/log.csv'
        self.file_w = open(self.file_name, 'w', newline='')
        self.writer = csv.writer(self.file_w)

        # write first line to csv
        self.keys = ['epoch'] + ['lr'] + [type(m).__name__ for m in self.eval_metrics]
        print(self.keys)
        self.writer.writerow(self.keys)

        # init all metrics with 0
        self.curr_stats = {
            type(m).__name__: 0.0 for m in self.eval_metrics
        }
        self.curr_stats['epoch'] = 0
        self.curr_stats['lr'] = self.optim.scheduler.lr

    def write_row(self):
        row = []
        for m in self.keys:
            row.append(self.curr_stats[m])

        self.writer.writerow(row)

    def log(self):
        pred = self.optim.preds / self.optim.size
        target = self.optim.targets / self.optim.size
        for metric in self.eval_metrics:
            score = metric(pred, target)
            self.curr_stats[type(metric).__name__] = score
        self.curr_stats['epoch'] = self.optim.scheduler.epoch
        self.curr_stats['lr'] = self.optim.scheduler.lr
        self.write_row()

    def plot(self):
        self.file_w.close()
        metric_dict = csv_to_dict_of_lists(self.file_name)

        # Convert values to floats
        iou_values = list(map(float, metric_dict['IoU']))
        hiou_values = list(map(float, metric_dict['HIoU']))
        ciou_values = list(map(float, metric_dict['CIoU']))

        # Plot the metrics
        plt.figure(figsize=(8, 5))
        plt.plot(iou_values, label='IoU')
        plt.plot(hiou_values, label='HIoU')
        plt.plot(ciou_values, label='CIoU')

        # Set fixed y-axis between 0 and 1 with ticks at 0.25 intervals
        plt.ylim(-1, 1)
        plt.yticks(np.arange(-1, 1.25, 0.25))
        plt.xlim(0, self.optim.scheduler.num_epochs + 1)

        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)  # Add a grid for clarity
        plt.xlabel("Epochs")
        plt.ylabel("Metric Value")
        plt.title("IoU Metrics Over Time")
        plt.savefig(f"./results/{self.name}/iou_metrics.png")

        plt.show(block=True)

    # destructor in python:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file_w.close()