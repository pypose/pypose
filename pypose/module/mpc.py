import sys
import torch
import pypose as pp
import torch.nn as nn


class MPC(nn.Module):

    def __init__(self, system, T, lqr_iter=1,
        not_improved_lim=5, best_cost_eps=1e-4):
        super().__init__()
        self.system = system
        self.T = T
        self.lqr_iter = lqr_iter
        self.not_improved_lim = not_improved_lim
        self.best_cost_eps = best_cost_eps

    def forward(self, x_init, Q, p, n_batch=1, u_init=None):
        
        for i in range(self.lqr_iter):
            _lqr = pp.module.LQR(self.system, Q, p, self.T, n_batch=n_batch)
            x, u, cost = _lqr(x_init)

        _lqr = pp.module.LQR(self.system, Q, p, self.T, n_batch=n_batch)
        x, u, cost = _lqr(x_init)

        return x, u, cost
