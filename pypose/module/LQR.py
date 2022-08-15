import torch as torch
import torch.nn as nn
import pypose as pp
from . import util

import numpy as np
import numpy.random as npr
from collections import namedtuple

QuadCost = namedtuple('QuadCost', 'C c')
LinDx = namedtuple('LinDx', 'F f')
LqrForOut = namedtuple(
    'lqrForOut',
    'objs full_du_norm alpha_du_norm mean_alphas costs'
)

class DP_LQR:

    def __init__(self, n_state, n_ctrl, T, linesearch_decay=0.2, max_linesearch_iter=10, 
                true_cost=None, true_dynamics=None, current_x=None, current_u=None, verbose=0):
        
        self.n_state = n_state
        self.n_ctrl = n_ctrl
        self.T = T
        self.linesearch_decay = linesearch_decay
        self.max_linesearch_iter = max_linesearch_iter
        self.true_cost = true_cost
        self.true_dynamics = true_dynamics
        self.current_x = current_x
        self.current_u = current_u
        self.verbose = verbose

    def DP_LQR_backward(self, C, c, F, f):

        Ks = []
        ks = []
        Vtp1 = vtp1 = None
        for t in range(self.T-1, -1, -1): #range(start, stop, step): the stop number itself is always omitted.
            if t == self.T-1:
                Qt = C[t]
                qt = c[t]
            else:
                Ft = F[t]
                Ft_T = Ft.mT
                Qt = C[t] + Ft_T.matmul(Vtp1).bmm(Ft)
                if f is None or f.numel() == 0: #alias for nelement(): count the number of elements of the tensor
                    qt = c[t] + Ft_T.matmul(vtp1) #Tensor multiplied Nonetype
                else:
                    ft = f[t]
                    qt = c[t] + Ft_T.matmul(Vtp1).matmul(ft) + Ft_T.matmul(vtp1)
            
            n_state = self.n_state
            Qt_xx = Qt[:, :n_state, :n_state]
            Qt_xu = Qt[:, :n_state, n_state:]
            Qt_ux = Qt[:, n_state:, :n_state]
            Qt_uu = Qt[:, n_state:, n_state:]
            qt_x = qt[:, :n_state]
            qt_u = qt[:, n_state:]

            if self.n_ctrl == 1:
                Kt = -(1./Qt_uu)*Qt_ux
                kt = -(1./Qt_uu.squeeze(2))*qt_u
            else:
                Qt_uu_inv = [torch.linalg.pinv(Qt_uu[i]) for i in range(Qt_uu.shape[0])]
                Qt_uu_inv = torch.stack(Qt_uu_inv) #Concatenates a sequence of tensors along a new dimension. All tensors need to be of the same size.
                Kt = -Qt_uu_inv.matmul(Qt_ux)
                kt = -Qt_uu_inv.matmul(qt_u)

            Kt_T = Kt.mT

            Ks.append(Kt)
            ks.append(kt)

            Vtp1 = Qt_xx + Qt_xu.bmm(Kt) + Kt_T.bmm(Qt_ux) + Kt_T.bmm(Qt_uu).bmm(Kt)
            vtp1 = qt_x + Qt_xu.matmul(kt) + Kt_T.matmul(qt_u) + Kt_T.bmm(Qt_uu).matmul(kt)

        return Ks, ks

    def DP_LQR_forward(self, x_init, C, c, F, f, Ks, ks):
        x = self.current_x
        u = self.current_u
        n_batch = C.size(1)

        old_cost = util.get_cost(self.T, u, self.true_cost, self.true_dynamics, x=x)

        current_cost = None
        alphas = torch.ones(n_batch).type_as(C)
        full_du_norm = None

        i =0
        while (current_cost is None or \
               (old_cost is not None and \
                torch.any((current_cost > old_cost)).cpu().item() == 1)) and \
                 i < self.max_linesearch_iter:
            new_u = []
            new_x = [x_init]
            dx = [torch.zeros_like(x_init)]
            objs = []
            for t in range(self.T):
                t_rev = self.T-1-t
                Kt = Ks[t_rev]
                kt = ks[t_rev]
                new_xt = new_x[t]
                xt = x[t]
                ut = u[t]
                dxt = dx[t]
                new_ut = Kt.matmul(dxt) + ut + torch.diag(alphas).mm(kt)
                new_u.append(new_ut)

                new_xut = torch.cat((new_xt, new_ut), dim=1)
                if t < self.T-1:
                    if isinstance(self.true_dynamics, LinDx):
                        F, f = self.true_dynamics.F, self.true_dynamics.f
                        new_xtp1 = F[t].matmul(new_xut)
                        if f is not None and f.numel() > 0:
                            new_xtp1 += f[t]
                    else:
                        new_xtp1 = self.true_dynamics(new_xt, new_ut)

                    new_x.append(new_xtp1)
                    dx.append(new_xtp1 - x[t+1])

                if isinstance(self.true_cost, QuadCost):
                    C, c = self.true_cost.C, self.true_cost.c
                    obj = 0.5*new_xut.matmul(C[t]).matmul(new_xut) + new_xut.matmul(c[t])
                else:
                    obj = self.true_cost(new_xut)
                objs.append(obj)

            objs = torch.stack(objs)
            current_cost = torch.sum(objs, dim=0)

            new_u = torch.stack(new_u)
            new_x = torch.stack(new_x)
            if full_du_norm is None:
                full_du_norm = (u-new_u).transpose(1,2).contiguous().view(
                    n_batch, -1).norm(2, 1)

            alphas[current_cost > old_cost] *= self.linesearch_decay
            i += 1

        alphas[current_cost > old_cost] /= self.linesearch_decay
        alpha_du_norm = (u-new_u).transpose(1,2).contiguous().view(
            n_batch, -1).norm(2, 1)

        return new_x, new_u, LqrForOut(
            objs, full_du_norm,
            alpha_du_norm,
            torch.mean(alphas),
            current_cost
        )
