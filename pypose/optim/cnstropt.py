import torch
from torch import nn
from .scndopt import _Optimizer


class AugModel(nn.Module):
    '''
    Internal Augmented Model class for Stochastic Augumented Largragian Optimization.
    '''
    def __init__(self, model, pf):
        super().__init__()
        self.model, self.pf = model, pf

    def update_lmd(self, error):
        self.lmd += error * self.pf
        return self.lmd

    def update_pf(self, pnf_update_step, safe_guard):
        pf = self.pf * pnf_update_step
        self.pf = pf if pf < safe_guard else safe_guard

    def forward(self, inputs=None, target=None):
        R, C = self.model(inputs)
        if not hasattr(self, 'lmd'):
            self.lmd = torch.zeros(C.shape[0], device=C.device, dtype=C.dtype)
        return R + self.lmd @ C + self.pf * C.norm()**2 / 2

############
    # Update Needed Parameters:
    #   1. model params: \theta, update with SGD
    #   2. lambda multiplier: \lambda, \lambda_{t+1} = \lambda_{t} + pf * error_C
    #   3. penalty factor(Optional): update_para * penalty factor
class SAL(_Optimizer):
    r'''
    Stochastic Augmented Lagrangian method for Equality Constrained Optimization.
    '''
    def __init__(self, model, optim, penalty_factor=1, penalty_safeguard=1e5, \
                       penalty_update_factor=2, decrease_rate=0.9, min=1e-6, max=1e32):
        defaults = {**{'min':min, 'max':max}}
        super().__init__(model.parameters(), defaults=defaults)
        self.model, self.optim = model, optim
        self.decrease_rate = decrease_rate
        self.pf_rate = penalty_update_factor
        self.pf_safeguard = penalty_safeguard
        self.alm_model = AugModel(self.model, pf=penalty_factor)
        self.inner_iter = 0
        self.best_violation = float('inf')
        self.last = float('inf')

    #### f(x) - y = loss_0, f(x) + C(x) - 0 - y
    def step(self, inputs=None):

        with torch.no_grad():
            self.last_object_value, cnst = self.model(inputs)
            self.last_violation = cnst.norm()

        for _ in range(self.inner_iter):
            self.optim.zero_grad()
            loss = self.alm_model(inputs)
            loss.backward()
            self.optim.step()

        with torch.no_grad():
            self.object_value, self.violation = self.model(inputs)
            self.violation_norm = torch.norm(self.violation)

            if self.violation_norm <= self.best_violation * self.decrease_rate:
                self.alm_model.update_lmd(self.violation)
                self.best_violation = self.violation_norm

            # if violation is not well satisfied, add further punishment
            else:
                self.alm_model.update_pf(self.pf_rate, self.pf_safeguard)

        return loss
