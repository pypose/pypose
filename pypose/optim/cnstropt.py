from __future__ import annotations

import torch
from torch import nn
from .scndopt import _Optimizer
from .scheduler import _Scheduler


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

        def update_params(self, inputs=None, target=None):
            loss, cnstr = self.model(inputs)
            violate = cnstr.norm()
            if violate <= self.violate * self.hedge:
                self.lmd += cnstr * self.penalty
                self.violate = violate
            else:
                self.penalty = min(self.penalty * self.scale, self.shield)
            return torch.tensor([loss, violate])

        def forward(self, inputs=None, target=None):
            R, C = self.model(inputs)
            if self.lmd is None:
                self.lmd = torch.zeros(C.shape[0], device=C.device, dtype=C.dtype)
            return R + self.lmd @ C + self.penalty * C.norm()**2 / 2

    def __init__(self, model, scheduler, steps:int=None,
                 penalty:float=1, shield:float=1e3, scale:float=2, hedge:float=0.9):
        super().__init__(model.parameters(), defaults={})
        assert issubclass(type(scheduler), _Scheduler) or steps is not None, \
            "SAL config has to be:" \
            "['scheduler' from torch and 'steps' not None] Or ['scheduler' from pypose]"
        self.auglag = self.AugLag(model, penalty, shield, scale, hedge)
        self.scheduler, self.steps = scheduler, steps
        self.last = self.loss = torch.tensor([float('inf'), float('inf')])

    def step(self, inputs=None):

        with torch.no_grad():
            if (self.last == float('inf')).all():
                last, violate = self.auglag.model(inputs)
                self.last = self.loss = torch.tensor([last, violate.norm()])

        scheduler = self.scheduler()
        if issubclass(type(scheduler), _Scheduler):
            # For optim from pypose
            while scheduler.continual():
                loss = scheduler.optimizer.step(inputs)
                scheduler.step(loss)
        else:
            # For optim from torch
            for _ in range(self.steps):
                scheduler.optimizer.zero_grad()
                loss = self.auglag(inputs)
                loss.backward()
                scheduler.optimizer.step()

        with torch.no_grad():
            self.last = self.loss
            self.loss = self.auglag.update_params(inputs)

        return self.loss
