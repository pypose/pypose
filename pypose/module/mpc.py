import torch
import pypose as pp
import torch.nn as nn


class MPC(nn.Module):

    def __init__(self, system, T, lqr_iter=1, best_cost_eps=1e-4):
        super().__init__()
        self.system = system
        self.T = T
        self.lqr_iter = lqr_iter
        self.best_cost_eps = best_cost_eps

    def forward(self, x_init, Q, p, u_init=None):
        
        best = None

        for i in range(self.lqr_iter):
            _lqr = pp.module.LQR(self.system, Q, p, self.T)
            x, u, cost = _lqr(x_init)
            costs = torch.sum(cost, dim=0)

            """ if best is None:
                best = {
                    'x': list(torch.split(x, split_size_or_sections=1, dim=1)),
                    'u': list(torch.split(u, split_size_or_sections=1, dim=1)),
                    'costs': costs,
                }
            else:
                if costs <= best['costs'] + self.best_cost_eps:
                    best['x'] = x.unsqueeze(1)
                    best['u'] = u.unsqueeze(1)
                    best['costs'] = costs

        x = torch.cat(best['x'], dim=1)
        u = torch.cat(best['u'], dim=1) """

        _lqr = pp.module.LQR(self.system, Q, p, self.T)
        x, u, cost = _lqr(x_init)

        return x, u, cost 
