import torch
from torch import nn
from .scndopt import _Optimizer


class AugModel(nn.Module):
    '''
    Internal Augmented Model class for Stochastic Augumented Largragian Optimization.
    '''
    def __init__(self, model, penalty):
        super().__init__()
        self.model, self.p = model, penalty

    def update_lmd(self, error):
        self.lmd += error * self.p
        return self.lmd

    def update_penalty(self, scale, safeguard):
        p = self.p * scale
        self.p = min(p, safeguard)

    def forward(self, inputs=None, target=None):
        R, C = self.model(inputs)
        if not hasattr(self, 'lmd'):
            self.lmd = torch.zeros(C.shape[0], device=C.device, dtype=C.dtype)
        return R + self.lmd @ C + self.p * C.norm()**2 / 2

############
    # Update Needed Parameters:
    #   1. model params: \theta, update with SGD
    #   2. Lagrangian multiplier: \lambda, \lambda_{t+1} = \lambda_{t} + pf * error_C
    #   3. penalty factor(Optional): update_para * penalty factor
class SAL(_Optimizer):
    r'''
    Stochastic Augmented Lagrangian method for Equality Constrained Optimization.
    '''
    def __init__(self, model, optim, penalty=1, safeguard=1e5, up=2, down=0.9):
        super().__init__(model.parameters(), defaults={})
        self.model, self.optim  = model, optim
        self.safeguard, self.up, self.down = safeguard, up, down
        self.augmod = AugModel(model, penalty=penalty)
        self.last = self.min_violate = float('inf')

    #### f(x) - y = loss_0, f(x) + C(x) - 0 - y
    def step(self, inputs=None):

        with torch.no_grad():
            self.last, cnstr = self.model(inputs)
            self.last_violation = cnstr.norm()

        for _ in range(self.inner_iter):
            self.optim.zero_grad()
            loss = self.augmod(inputs)
            loss.backward()
            self.optim.step()

        with torch.no_grad():
            self.loss, self.cnstr = self.model(inputs)
            self.violate = self.cnstr.norm()

            if self.violate <= self.min_violate * self.down:
                self.augmod.update_lmd(self.cnstr)
                self.min_violate = self.violate
            else:
                self.augmod.update_penalty(self.up, self.safeguard)

        return loss
