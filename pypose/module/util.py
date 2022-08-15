import torch as torch
import torch.nn as nn
import pypose as pp
from collections import namedtuple

QuadCost = namedtuple('QuadCost', 'C c')
LinDx = namedtuple('LinDx', 'F f')

def get_traj(T, u, x_init, dynamics):

    if isinstance(dynamics, LinDx):
        F = dynamics.F
        f = dynamics.f
        if f is not None:
            assert f.shape == F.shape[:3]

    x = x_init
    for t in range(T):
        xt = x[t]
        ut = u[t]
        if t < T-1:
            if isinstance(dynamics, LinDx):
                xut = torch.cat((xt, ut), 1)
                new_x = F[t].matmul(xut)
                if f is not None:
                    new_x += f[t]
            else:
                new_x = dynamics(xt,ut)
            x.append(new_x)
    x = torch.stack(x, dim=0)
    return x

def get_cost(T, u, cost, dynamics=None, x_init=None, x=None):
    assert x_init is not None or x is not None

    if isinstance(cost, QuadCost):
        C = cost.C
        c = cost.c

    if x is None:
        x = get_traj(T, u, x_init, dynamics)

    objs = []
    for t in range(T):
        xt = x[t]
        ut = u[t]
        xut = torch.cat((xt, ut), 1)
        if isinstance(cost, QuadCost):
            obj = 0.5*xut.matmul(C[t]).matmul(xut) + xut.matmul(c[t])
        else:
            obj = cost(xut)
        objs.append(obj)
    objs = torch.stack(objs, dim=0)
    total_obj = torch.sum(objs, dim=0)
    return total_obj