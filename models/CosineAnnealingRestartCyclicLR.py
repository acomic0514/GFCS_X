#!/usr/bin/env python
# coding: utf-8

# In[17]:


import math
import torch.optim.lr_scheduler as lr_scheduler

class CosineAnnealingRestartCyclicLR(lr_scheduler._LRScheduler):
    """ 自訂 Cosine Annealing Restart Cyclic LR """

    def __init__(self, optimizer, periods, restart_weights=[1], eta_mins=[1e-6], last_epoch=-1):
        assert len(periods) == len(restart_weights) == len(eta_mins), \
            "periods, restart_weights, and eta_mins must have the same length"
        
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_mins = eta_mins
        self.cumulative_period = [sum(self.periods[:i + 1]) for i in range(len(self.periods))]

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """ 計算當前學習率 """
        idx = next(i for i, v in enumerate(self.cumulative_period) if self.last_epoch <= v)

        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        current_period = self.periods[idx]
        eta_min = self.eta_mins[idx]

        return [
            eta_min + current_weight * 0.5 * (base_lr - eta_min) *
            (1 + math.cos(math.pi * ((self.last_epoch - nearest_restart) / current_period)))
            for base_lr in self.base_lrs
        ]


# In[ ]:




