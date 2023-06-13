import torch
import pypose as pp
import torch.nn as nn
from ..utils.stepper import ReduceToBason

class MPC(nn.Module):

    def __init__(self, system, T, step=10, eps=1e-4):
        super().__init__()
        self.system = system
        self.T = T
        self.step = step
        self.eps = eps

    def forward(self, Q, p, x_init, dt, current_x = None, current_u=None):

        best = None
        x = current_x
        u = current_u
        for i in range(self.step):
            if u is not None:
                u = u.detach()
            lqr = pp.module.LQR(self.system, Q, p, self.T)
            x, u, cost = lqr(x_init, dt, x, u)
            assert x.ndim == u.ndim == 3
            if best is None:
                best = {
                    'x': x,
                    'u': u,
                    'cost': cost,}
            else:
                if cost <= best['cost']+ self.eps:
                    best['x']= x
                    best['u']= u
                    best['cost'] = cost
        if self.step > 1:
            current_x = best['x']
            current_u = best['u']
        _lqr = pp.module.LQR(self.system, Q, p, self.T)
        x, u, cost= _lqr(x_init, dt, current_x, current_u)
        return x, u, cost
