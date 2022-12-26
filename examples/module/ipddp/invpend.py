import torch as torch
import torch.nn as nn
import pypose as pp
from pypose.module.dynamics import System
from pypose.module.ipddp import ddpOptimizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create class for inverted-pendulum dynamics
class InvPend(System):
    def __init__(self, dt):
        super(InvPend, self).__init__()
        self.tau = dt

    def state_transition(self, state, input, t=None):
        # x, xDot = state
        force = input.squeeze()
        # _dstate = torch.stack((state[0,1], force+torch.sin(state[0,0])))
        _dstate = torch.stack((state[0,1], force+state[0,0]))
        return state + torch.mul(_dstate, self.tau)

    def observation(self, state, input, t=None):
        return state

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    # Create parameters for inv pendulum trajectory
    dt = 0.05   # Delta t
    N = 10    # Number of time steps

    # Initial state
    state = torch.tensor([[-1., 0.]])

    # Create dynamics solver object
    sys = InvPend(dt)    # Calculate trajectory
    n_state = 2
    n_input = 1 
    # Calculate trajectory
    state_all =      torch.zeros(N+1, 1, n_state)
    input_all = 0.02*torch.ones(N,    1, n_input)
    init_traj = {'state': state_all, 
                 'input': input_all}
    state_all[0] = state
 

    # Create cost object
    Q = dt*torch.eye(n_state, n_state)
    R = dt*torch.eye(n_input, n_input)
    S = torch.zeros(n_state, n_input)
    c = torch.zeros(1, 1)
    stage_cost = pp.module.QuadCost(Q, R, S, c)
    terminal_cost = pp.module.QuadCost(10./dt*Q, R, S, c)

    # Create constraint object
    gx = torch.zeros( 2*n_input, n_state)
    gu = torch.vstack( (torch.eye(n_input, n_input), - torch.eye(n_input, n_input)) )
    g = torch.hstack( (-0.25 * torch.ones(1, n_input), -0.25 * torch.ones(1, n_input)) )
    lincon = pp.module.LinCon(gx, gu, g)
    solver = ddpOptimizer(sys, stage_cost, terminal_cost, lincon, n_state, n_input, gx.shape[0], N, init_traj) 

    traj_opt = solver.optimizer()


    