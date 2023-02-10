import sys
import torch
import pypose as pp
import torch.nn as nn


class MPC(nn.Module):

    def __init__(self, system, u_init = None, lqr_iter=1,
        not_improved_lim=5, best_cost_eps=1e-4):
        super().__init__()
        self.system = system
        self.u_init = u_init
        self.lqr_iter = lqr_iter
        self.not_improved_lim = not_improved_lim
        self.best_cost_eps = best_cost_eps

    def forward(self, x_init, Q, p):
        n_batch, T, ns, nc = p.size(-3), p.size(-2), self.system.B.size(-2), self.system.B.size(-1)
        nsc = ns + nc

        if Q.ndim != 4 or p.ndim != 3:
            print('MPC Error: Unexpected QuadCost shape.')
            sys.exit(-1)

        if self.u_init is None:
            u = torch.zeros(n_batch, T, nc).type_as(x_init.data)
        else:
            u = self.u_init
            if u.ndim == 2:
                u = u.unsqueeze(1).expand(n_batch, T, -1).clone()
        u = u.type_as(x_init.data)

        best = None

        n_not_improved = 0
        for i in range(self.lqr_iter):
            if u is not None or not u.requires_grad:
                u = u
            else:
                u = u.detach()
            u.requires_grad = False 

            _lqr = pp.module.LQR(system = self.system)
            x, u, cost = _lqr(x_init, Q, p)
            n_not_improved += 1
            assert x.ndim == u.ndim == 3
            costs = torch.sum(cost, dim=0)

            """ if best is None: 
                best = {
                    'x': list(torch.split(x, split_size_or_sections=1, dim=1)),
                    'u': list(torch.split(u, split_size_or_sections=1, dim=1)),
                    'costs': costs,
                }
            else:
                for j in range(n_batch):
                    if costs[j] <= best['costs'][j] + self.best_cost_eps:
                        n_not_improved = 0
                        best['x'][j] = x[:,j].unsqueeze(1)
                        best['u'][j] = u[:,j].unsqueeze(1)
                        best['costs'][j] = costs[j]

            if n_not_improved > self.not_improved_lim:
                break

        x = torch.cat(best['x'], dim=1)
        u = torch.cat(best['u'], dim=1) """

        _lqr = pp.module.LQR(system = self.system)
        x, u, cost= _lqr(x_init, Q, p)
        #costs = best['costs']
        
        return x, u, cost
