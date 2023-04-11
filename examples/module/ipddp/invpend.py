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
        force = input.squeeze()
        _dstate = torch.stack([state[...,1], force+self.gravity/self.length[0]*torch.sin(state[...,0].clone())], dim=-1)
        return state + torch.mul(_dstate, self.tau)

    def observation(self, state, input, t=None):
        return state

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    # Create parameters for inv pendulum trajectory
    dt = 0.05   # Delta t
    N = 5    # Number of time steps

    # Initial state
    state = torch.tensor([[-2.,0.],
                          [-1., 0.],
                          [-2.5, 1.]])

    sys = InvPend(dt) 
    n_state = 2
    n_input = 1 
    n_batch = 3
    state_all =      torch.zeros(n_batch, N+1, n_state)
    input_all = 0.02*torch.ones(n_batch,  N,   n_input)
    init_traj = {'state': state_all, 
                 'input': input_all}
    state_all[...,0,:] = state
 
    # Create cost object
    Q = torch.tile(dt*torch.eye(n_state, n_state, device=device), (n_batch, 1, 1))
    R = torch.tile(dt*torch.eye(n_input, n_input, device=device), (n_batch, 1, 1))
    S = torch.tile(torch.zeros(n_state, n_input, device=device), (n_batch, 1, 1))
    c = torch.tile(torch.zeros(1, 1, device=device), (n_batch, 1, 1))
    stage_cost = pp.module.QuadCost(Q, R, S, c)
    terminal_cost = pp.module.QuadCost(10./dt*Q, R, S, c)

    # Create constraint object
    gx = torch.zeros( 2*n_input, n_state)
    gu = torch.vstack( (torch.eye(n_input, n_input), - torch.eye(n_input, n_input)) )
    g = torch.hstack( (-0.25 * torch.ones(1, n_input), -0.25 * torch.ones(1, n_input)) )
    lincon = pp.module.LinCon(gx, gu, g)

    traj_opt = [None for batch_id in range(n_batch)]

    for batch_id in range(n_batch): # use for loop and keep the ddpOptimizer 
        stage_cost = pp.module.QuadCost(Q[batch_id], R[batch_id], S[batch_id], c[batch_id])
        terminal_cost = pp.module.QuadCost(10./dt*Q[batch_id], R[batch_id], S[batch_id], c[batch_id])
        lincon = pp.module.LinCon(gx, gu, g)  
        init_traj_sample = {'state': torch.unsqueeze(init_traj['state'][batch_id],1), 
                            'input': torch.unsqueeze(init_traj['input'][batch_id],1)} 
        solver = ddpOptimizer(sys, stage_cost, terminal_cost, lincon, n_state, n_input, gx.shape[-2], N, init_traj_sample) 
        traj_opt[batch_id] = solver.optimizer()


    