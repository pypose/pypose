""" import torch
import torch.nn as nn
import sys
sys.path.append("..")
import pypose as pp
import time

class MPC(nn.Module):

    def __init__(self, n_state, n_ctrl, T, system, u_init = None, lqr_iter=1, n_batch = None,
        not_improved_lim=5, best_cost_eps=1e-4):
        
        super().__init__()
 
        self.n_state = n_state
        self.n_ctrl = n_ctrl
        self.T = T
        self.system = system
        self.u_init = u_init
        self.lqr_iter = lqr_iter
        self.n_batch = n_batch
        self.not_improved_lim = not_improved_lim
        self.best_cost_eps = best_cost_eps

    def forward(self, x_init, Q, p):

        if self.n_batch is not None:
            n_batch = self.n_batch
        elif Q.ndim == 4:
            n_batch = Q.size(0)
        else:
            print('MPC Error: Could not infer batch size, pass in as n_batch')
            sys.exit(-1)

        if Q.ndim == 2:
            Q = Q.unsqueeze(0).unsqueeze(0).expand(
                n_batch, self.T, self.n_state+self.n_ctrl, -1)
        elif Q.ndim == 3:
            Q = Q.unsqueeze(0).expand(
                n_batch, self.T, self.n_state+self.n_ctrl, -1)

        if p.ndim == 1:
            p = p.unsqueeze(0).unsqueeze(0).expand(n_batch, self.T, -1)
        elif p.ndim == 2:
            p = p.unsqueeze(0).expand(n_batch, self.T, -1)

        if Q.ndim != 4 or p.ndim != 3:
            print('MPC Error: Unexpected QuadCost shape.')
            sys.exit(-1)

        if self.u_init is None:
            u = torch.zeros(n_batch, self.T, self.n_ctrl).type_as(x_init.data)
        else:
            u = self.u_init
            if u.ndim == 2:
                u = u.unsqueeze(1).expand(n_batch, self.T, -1).clone()
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
            x, u, costs = _lqr(x_init, Q, p)
            #costs = torch.sum(objs, dim=0)

            n_not_improved += 1

            assert x.ndim == 3
            assert u.ndim == 3


        _lqr = pp.module.LQR(system = self.system)
        x, u, costs = _lqr(x_init, Q, p)

        return x, u, costs """


import torch
import torch.nn as nn
import sys
sys.path.append("..")
import pypose as pp

class MPC(nn.Module):

    def __init__(self, n_state, n_ctrl, T, system, u_init = None, lqr_iter=1, n_batch = None,
        not_improved_lim=5, best_cost_eps=1e-4):
        
        super().__init__()
 
        self.n_state = n_state
        self.n_ctrl = n_ctrl
        self.T = T
        self.system = system
        self.u_init = u_init
        self.lqr_iter = lqr_iter
        self.n_batch = n_batch
        self.not_improved_lim = not_improved_lim
        self.best_cost_eps = best_cost_eps

    def forward(self, x_init, Q, p):

        if self.n_batch is not None:
            n_batch = self.n_batch
        elif Q.ndim == 4:
            n_batch = Q.size(0)
        else:
            print('MPC Error: Could not infer batch size, pass in as n_batch')
            sys.exit(-1)

        if Q.ndim == 2:
            Q = Q.unsqueeze(0).unsqueeze(0).expand(
                n_batch, self.T, self.n_state+self.n_ctrl, -1)
        elif Q.ndim == 3:
            Q = Q.unsqueeze(0).expand(
                n_batch, self.T, self.n_state+self.n_ctrl, -1)

        if p.ndim == 1:
            p = p.unsqueeze(0).unsqueeze(0).expand(n_batch, self.T, -1)
        elif p.ndim == 2:
            p = p.unsqueeze(0).expand(n_batch, self.T, -1)

        if Q.ndim != 4 or p.ndim != 3:
            print('MPC Error: Unexpected QuadCost shape.')
            sys.exit(-1)

        if self.u_init is None:
            u = torch.zeros(n_batch, self.T, self.n_ctrl).type_as(x_init.data)
        else:
            u = self.u_init
            if u.ndim == 2:
                u = u.unsqueeze(1).expand(n_batch, self.T, -1).clone()
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
            x, u, costs= _lqr(x_init, Q, p)
            costs = torch.sum(costs, dim=0)

            n_not_improved += 1

            assert x.ndim == 3
            assert u.ndim == 3

            if best is None: 
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
        u = torch.cat(best['u'], dim=1)

        _lqr = pp.module.LQR(system = self.system)
        x, u, costs= _lqr(x_init, Q, p)

        costs = best['costs']
        return x, u, costs

