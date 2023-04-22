import pypose as pp
import torch as torch
import torch.nn as nn
from pypose.module.dynamics import System, NLS
from pypose.module.ipddp import ddpOptimizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create class for inverted-pendulum dynamics
class InvPend(NLS): # why use NLS to inherit?
    def __init__(self, dt, length=[10.0], gravity=10.0):
        super(InvPend, self).__init__()
        self.tau = dt
        self.length = length
        self.gravity = gravity

    def state_transition(self, state, input, t=None):
        force = input.squeeze(-1)
        _dstate = torch.stack([state[...,1], force+self.gravity/self.length[0]*torch.sin(state[...,0].clone())], dim=-1)
        return state + torch.mul(_dstate, self.tau)

    def observation(self, state, input, t=None):
        return state

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    # Create parameters for inv pendulum trajectory
    dt = 0.05   # Delta t
    T = 5    # Number of time steps

    # Initial state
    state = torch.tensor([[-2.,0.],
                          [-1., 0.],
                          [-2.5, 1.]])

    sys = InvPend(dt) 
    ns, nc = 2, 1
    n_batch = 3
    state_all =      torch.zeros(n_batch, T+1, ns)
    input_all = 0.02*torch.ones(n_batch,  T,   nc)
    init_traj = {'state': state_all, 
                 'input': input_all}
    state_all[...,0,:] = state
 
    # Create cost object
    Q = torch.tile(dt*torch.eye(ns, ns, device=device), (n_batch, T, 1, 1))
    R = torch.tile(dt*torch.eye(nc, nc, device=device), (n_batch, T, 1, 1))
    S = torch.tile(torch.zeros(ns, nc, device=device), (n_batch, T, 1, 1))
    c = torch.tile(torch.zeros(1, device=device), (n_batch, T))
    stage_cost = pp.module.QuadCost(Q, R, S, c)
    terminal_cost = pp.module.QuadCost(10./dt*Q[...,0:1,:,:], R[...,0:1,:,:], S[...,0:1,:,:], c[...,0:1]) # special stagecost with T=1

    # Create constraint object
    gx = torch.tile(torch.zeros( 2*nc, ns), (n_batch, T, 1, 1))
    gu = torch.tile(torch.vstack( (torch.eye(nc, nc), - torch.eye(nc, nc)) ), (n_batch, T, 1, 1))
    g = torch.tile(torch.hstack( (-0.25 * torch.ones(nc), -0.25 * torch.ones(nc)) ), (n_batch, T, 1))
    lincon = pp.module.LinCon(gx, gu, g)

    traj_opt = [None for batch_id in range(n_batch)]

    for batch_id in range(n_batch): # use for loop and keep the ddpOptimizer 
        # used class is for batched version, inside loop use batch_size = 1
        stage_cost = pp.module.QuadCost(Q[batch_id:batch_id+1], R[batch_id:batch_id+1], S[batch_id:batch_id+1], c[batch_id:batch_id+1])
        terminal_cost = pp.module.QuadCost(10./dt*Q[batch_id:batch_id+1,0:1,:,:], R[batch_id:batch_id+1,0:1,:,:], S[batch_id:batch_id+1,0:1,:,:], c[batch_id:batch_id+1,0:1])
        lincon = pp.module.LinCon(gx[batch_id:batch_id+1], gu[batch_id:batch_id+1], g[batch_id:batch_id+1])  
        init_traj_sample = {'state': init_traj['state'][batch_id:batch_id+1], 
                            'input': init_traj['input'][batch_id:batch_id+1]} 
        solver = ddpOptimizer(sys, stage_cost, terminal_cost, lincon, ns, nc, gx.shape[-2], T, init_traj_sample) 
        traj_opt[batch_id] = solver.optimizer()


    