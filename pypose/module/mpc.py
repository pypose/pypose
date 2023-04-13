import torch
import pypose as pp
import torch.nn as nn


class MPC(nn.Module):

    def __init__(self, system, T, step=10, eps=1e-4):
        super().__init__()
        self.system = system
        self.T = T
        self.step = step
        self.eps = eps

    def forward(self, Q, p, x_init, current_x=None, current_u=None, time=None):

        best = None

        for i in range(self.step):
            if current_x is not None:
                current_u.requires_grad = False
                if not current_u.requires_grad:
                    current_u = current_u
                else:
                    current_u = current_u.detach()

            lqr = pp.module.LQR(self.system, Q, p, self.T)
            x, u, cost = lqr(x_init, current_x, current_u, time)
            assert x.ndim == u.ndim == 3

            if best is None:
                best = {
                    'x': x,
                    'u': u,
                    'cost': cost,}
            else:
                if cost <= best['cost']+ self.eps:
                    best['x'] = x
                    best['u']= u
                    best['cost'] = cost

            current_x = x
            current_u = u

        if self.step == 1:
            current_x = None
            current_u = None
        else:
            current_x = best['x']
            current_u = best['u']

        _lqr = pp.module.LQR(self.system, Q, p, self.T)
        x, u, cost= _lqr(x_init, current_x, current_u, time)

        return x, u, cost
