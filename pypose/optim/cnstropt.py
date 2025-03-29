from __future__ import annotations

import torch
from torch import nn
from ..utils import Prepare
from .scheduler import _Scheduler
from .scndopt import _Optimizer, PArgs
from torch.optim.lr_scheduler import LRScheduler


class SAL(_Optimizer):
    r'''
    Stochastic Augmented Lagrangian method for Equality Constrained Optimization.
    '''
    class AugLag(nn.Module):
        '''
        Internal Augmented Largragian Model for Equality Constrained Optimization.
        The input model has to output a 2-tuple, where first element is a scalar denoting
        the loss, and the second is a vector denoting the a series of equality violations.
        '''
        def __init__(self, model, penalty, shield, scale, hedge):
            super().__init__()
            self.model, self.lmd, self.penalty, self.shield = model, None, penalty, shield
            self.scale, self.hedge, self.violate = scale, hedge, float('inf')

        def update_params(self, *args, **kwargs):
            loss, cnstr = self.model(*args, **kwargs)
            violate = cnstr.norm()
            if violate <= self.violate * self.hedge:
                self.lmd += cnstr * self.penalty
                self.violate = violate
            else:
                self.penalty = min(self.penalty * self.scale, self.shield)
            return torch.tensor([loss, violate])

        def forward(self, *args, **kwargs):
            R, C = self.model(*args, **kwargs)
            if self.lmd is None:
                self.lmd = torch.zeros(C.shape[0], device=C.device, dtype=C.dtype)
            return R + self.lmd @ C + self.penalty * C.norm()**2 / 2

    def __init__(self, model, penalty=1, shield=1e3, scale=2, hedge=0.9):
        super().__init__(model.parameters(), defaults={})
        self.model = self.AugLag(model, penalty, shield, scale, hedge)
        self.last = self.loss = torch.tensor([float('inf'), float('inf')])
        self.scheduler = None

    def inner_scheduler(self, scheduler:Prepare, steps:int=None):
        r'''
        Set up a Prepare class for an inner scheduler.
        '''
        assert not (issubclass(scheduler.class_name, LRScheduler) and steps is None), \
            "steps is needed if scheduler is a torch scheduler."
        self.scheduler, self.steps = scheduler, steps

    def step(self, *args, **kwargs):
        assert self.scheduler is not None, \
            ".inner_scheduler() should be called before .step()"

        with torch.no_grad():
            if (self.last == float('inf')).all():
                last, violate = self.model.model(*args, **kwargs)
                self.last = self.loss = torch.tensor([last, violate.norm()])

        scheduler = self.scheduler()
        if issubclass(type(scheduler), _Scheduler):
            # For optim from pypose
            while scheduler.continual():
                loss = scheduler.optimizer.step(PArgs(*args, **kwargs))
                scheduler.step(loss)
        else:
            # For optim from torch
            for _ in range(self.steps):
                scheduler.optimizer.zero_grad()
                loss = self.model(*args, **kwargs)
                loss.backward()
                scheduler.optimizer.step()

        with torch.no_grad():
            self.last = self.loss
            self.loss = self.model.update_params(*args, **kwargs)

        return self.loss
