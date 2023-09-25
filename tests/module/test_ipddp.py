import pypose as pp
import torch as torch
import torch.nn as nn
from pypose.module.ipddp import IPDDP
from pypose.module.dynamics import NLS
from examples.module.dynamics.invpend import InvPend

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    # Create parameters for inv pendulum trajectory
    dt = 0.05   # Delta t
    T = 5    # Number of time steps

    # Initial state
    state = torch.tensor([
            [-2.,0.],
            [-1., 0.],
            [-2.5, 1.]],
            device=device)

    sys = InvPend(dt)
    ns, nc, n_batch = sys.xdim, sys.udim, state.shape[0]
    input_all = torch.ones(n_batch, T, nc, device=device) * 0.02
    # input of an infeasible traj test
    # input_all = torch.ones(n_batch, T, nc, device=device) * 0.02 + 0.25

    # Create cost object
    Q = torch.tile(dt * torch.eye(ns, ns, device=device), (n_batch, T, 1, 1))
    R = torch.tile(dt * torch.eye(nc, nc, device=device), (n_batch, T, 1, 1))
    S = torch.tile(torch.zeros(ns, nc, device=device), (n_batch, T, 1, 1))
    c = torch.tile(torch.zeros(1, device=device), (n_batch, T))
    stage_cost = pp.module.QuadCost(Q, R, S, c)
    terminal_cost = pp.module.QuadCost(10./dt*Q[...,0:1,:,:], R[...,0:1,:,:], S[...,0:1,:,:], c[...,0:1]) # special stagecost with T=1

    # Create constraint object
    gx = torch.tile(torch.zeros(2 * nc, ns, device=device), (n_batch, T, 1, 1))
    gu = torch.tile(torch.vstack((torch.eye(nc, nc, device=device), - torch.eye(nc, nc, device=device)) ), (n_batch, T, 1, 1))
    g = torch.tile(torch.hstack((- 0.25 * torch.ones(nc, device=device), - 0.25 * torch.ones(nc, device=device)) ), (n_batch, T, 1))
    lincon = pp.module.LinCon(gx, gu, g)

    traj_opt = [None for batch_id in range(n_batch)]

    for batch_id in range(n_batch): # use for loop and keep the IPDDP
        # used class is for batched version, inside loop use batch_size = 1
        stage_cost = pp.module.QuadCost(Q[batch_id:batch_id+1], R[batch_id:batch_id+1], S[batch_id:batch_id+1], c[batch_id:batch_id+1])
        terminal_cost = pp.module.QuadCost(10./dt*Q[batch_id:batch_id+1,0:1,:,:], R[batch_id:batch_id+1,0:1,:,:], S[batch_id:batch_id+1,0:1,:,:], c[batch_id:batch_id+1,0:1])
        lincon = pp.module.LinCon(gx[batch_id:batch_id+1], gu[batch_id:batch_id+1], g[batch_id:batch_id+1])
        ipddp = IPDDP(sys, stage_cost, terminal_cost, lincon, T, B=(1,))
        x_init, u_init = state[batch_id:batch_id+1], input_all[batch_id:batch_id+1]
        traj_opt[batch_id] = ipddp.solver(x_init, u_init=None, verbose=True)
