import torch.optim
import numpy as np

class CustomWarmupCosineScheduler:
    def __init__(self, optimizer: torch.optim, total_epochs, warmup1=10, anneal1=40, warmup2=10, anneal2=40,
                 lr_max1=0.0003, lr_min1=0.00005, lr_max2=0.00015, lr_min2=0):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.warmup1 = warmup1
        self.anneal1 = anneal1
        self.warmup2 = warmup2
        self.anneal2 = anneal2
        self.lr_max1 = lr_max1
        self.lr_min1 = lr_min1
        self.lr_max2 = lr_max2
        self.lr_min2 = lr_min2

    def step(self, epoch):
        if epoch < self.warmup1:
            # First warmup: linear from 0 to lr_max1
            lr = self.lr_max1 * (epoch + 1) / self.warmup1
        elif epoch < self.warmup1 + self.anneal1:
            # First cosine annealing: lr_max1 to lr_min1
            progress = (epoch - self.warmup1) / self.anneal1
            lr = self.lr_min1 + 0.5 * (self.lr_max1 - self.lr_min1) * (1 + np.cos(np.pi * progress))
        elif epoch < self.warmup1 + self.anneal1 + self.warmup2:
            # Second warmup: linear from lr_min1 to lr_max2
            progress = (epoch - self.warmup1 - self.anneal1 + 1) / self.warmup2
            lr = self.lr_min1 + (self.lr_max2 - self.lr_min1) * progress
        else:
            # Second cosine annealing: lr_max2 to lr_min2
            progress = (epoch - self.warmup1 - self.anneal1 - self.warmup2) / self.anneal2
            lr = self.lr_min2 + 0.5 * (self.lr_max2 - self.lr_min2) * (1 + np.cos(np.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
