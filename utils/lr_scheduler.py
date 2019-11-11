"""Popular Learning Rate Schedulers"""
from __future__ import division
import math
import torch

from bisect import bisect_right

__all__ = ['IterationPolyLR']

class IterationPolyLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, target_lr=0, max_iters=0, power=0.9, last_epoch=-1):
        self.target_lr = target_lr
        self.max_iters = max_iters
        self.power = power
        super(IterationPolyLR, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        N = self.max_iters 
        T = self.last_epoch
        factor = pow(1 - T / N, self.power)
        # https://blog.csdn.net/mieleizhi0522/article/details/83113824
        return [self.target_lr + (base_lr - self.target_lr) * factor for base_lr in self.base_lrs]

