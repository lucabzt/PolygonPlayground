"""
This class is responsible for computing the gradients and regressing towards the target polygon
"""
from typing import Callable

import torch
from torch import nn


class Optimizer:
    def __init__(self, loss_fn, preds, targets, size = 500, lr=0.1, num_epochs=200, logger=None, loss_steps=None):
        self.loss_steps = [0.8, 0.9] if loss_steps is None else loss_steps
        self.size = size
        self.loss_fn = loss_fn
        self.scheduler = Scheduler(lr, self.loss_steps, num_epochs)
        self.preds = preds
        self.targets = targets
        self.logger = logger
        assert self.preds.dim() == 3 and self.targets.dim() == 3, "Shape of pred and target must be [B, N, 2]. Forgot batch dimension?"


    def loss_one_sample(self, pred, target):
        pred = pred.unsqueeze(0).float()
        target = target.unsqueeze(0).float()
        return self.loss_fn(pred, target)


    def compute_gradients(self) -> torch.Tensor:
        gradients = []
        # use normalized preds and targets for loss calculation
        pred_nm = self.preds / self.size
        target_nm = self.targets / self.size
        for idx, pred in enumerate(pred_nm):
            pred = pred.requires_grad_()
            loss = self.loss_one_sample(pred, target_nm[idx])
            grad = torch.autograd.grad(loss, pred, retain_graph=False)[0]
            gradients.append(grad)

        return torch.concat(gradients, dim=0)


    def step(self) -> None:
        gradients = self.compute_gradients()
        gradients_lr = self.scheduler.lr * gradients * self.size
        # perform gradient descent
        with torch.no_grad():
            self.preds = self.preds - gradients_lr
        self.scheduler.step()
        if self.logger:
            self.logger.log()

    def n_steps(self, n):
        for i in range(n):
            self.step()

class Scheduler:
    def __init__(self, lr, loss_steps, num_epochs=200):
        self.init_lr = lr
        self.lr = lr
        self.num_epochs = num_epochs
        self.epoch = 0
        self.loss_steps = loss_steps
        assert len(loss_steps) == 2, f"You need to provide more loss steps. {loss_steps}"

    def step(self):
        self.epoch += 1
        if self.epoch > self.loss_steps[1] * self.num_epochs:
            self.lr = self.init_lr * 0.01
            return
        if self.epoch > self.loss_steps[0] * self.num_epochs:
            self.lr = self.init_lr * 0.1
            return
        return