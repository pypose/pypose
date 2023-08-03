import torch
import pypose
import math
import torch as th
import numpy as np
import pypose as pp
from torch import nn
import matplotlib.pyplot as plt
from torch.autograd.functional import jacobian
import pypose.optim.functional as F
from pypose.lietensor import randn_SE3
from pypose.optim.solver import Cholesky
from pypose.optim.strategy import Constant
from pypose.optim.scheduler import StopOnPlateau
from pypose.optim.optimizer import LevenbergMarquardt as LM
from pypose.optim.alm_optimizer import Augmented_Lagrangian_Algorithm as ALM
from scipy.spatial.transform import Rotation
from torch import matmul as mult

class LQR_Solver(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, A, B, C, c, T, x0):
        # Set number of inputs and outputs
        n_state = A.size(0)
        n_input = B.size(1)
        n_all = n_state + n_input

        # Construct cost matrix
        cost_mat = torch.block_diag(*C)

        # Construct dynamics solver matrix
        AB = torch.concat((A, B), dim = 1).unsqueeze(0).repeat(T - 1, 1, 1)
        AB = torch.block_diag(*AB)
        dynamics = torch.zeros(n_state * T, n_all * T)
        dynamics[:n_state,:n_state] = torch.eye(n_state)
        dynamics[n_state:n_state + AB.size(0), :AB.size(1)] = AB
        idx1c = torch.linspace(n_all, n_all * (T - 1), T - 1, dtype=int)
        idx1r = n_state * torch.linspace(1, T - 1, T - 1, dtype=int)
        for i in range(0,n_state):
            dynamics[idx1r + i, idx1c + i] = -1

        # Create full matrix
        zero_mat = torch.zeros(dynamics.size(0), dynamics.size(0))
        full_mat = torch.cat((torch.cat((cost_mat, dynamics.transpose(0, 1)), dim = 1),
                              torch.cat((dynamics, zero_mat),                 dim = 1)), dim = 0)

        # Create solution vector
        sol = torch.zeros(A.size(0) * T)
        sol[:n_state] = x0
        sol = torch.cat((torch.zeros(cost_mat.size(0)), sol), dim = 0).unsqueeze(-1)

        # Final solution
        tau_mu = torch.linalg.solve(full_mat, sol).squeeze()

        tau_star = tau_mu[:dynamics.size(1)]
        mu_star = tau_mu[dynamics.size(1):]

        return tau_star, mu_star

class PoseNet(nn.Module):
    def __init__(self, T, C, c, n_all):
        super().__init__()
        self.input = torch.nn.Parameter(torch.randn(T, n_all).flatten())
        self.C = C
        self.c = c

    def forward(self, input=None):
        cost = 0.5 * mult(mult(self.input, torch.block_diag(*self.C)), self.input)
        return cost

class ConstrainNet(nn.Module):
    def __init__(self, A, B, T, net,x0):
        super().__init__()
        self.A = A
        self.B = B
        self.T = T
        self.net = net
        self.x0 = x0
        
    def forward(self,input=None):
        # Calculate constraints
        n_state = self.A.size(0)
        n_input = self.B.size(1)
        n_all = n_state + n_input

        AB = torch.concat((self.A, self.B), dim = 1).unsqueeze(0).repeat(self.T - 1, 1, 1)
        AB = torch.block_diag(*AB)
        dynamics = torch.zeros(n_state * self.T, n_all * self.T)
        dynamics[:n_state, :n_state] = torch.eye(n_state)
        dynamics[n_state: n_state + AB.size(0), :AB.size(1)] = AB
        idx1c = torch.linspace(n_all, n_all * (self.T - 1), self.T - 1, dtype = int)
        idx1r = n_state * torch.linspace(1, self.T - 1, self.T - 1, dtype = int)
        for i in range(0, n_state):
            dynamics[idx1r + i, idx1c + i] = -1

        b = torch.zeros(dynamics.size(0))
        b[:n_state] = self.x0

        return mult(dynamics, self.net.input) - b

def test_optim_liegroup():
    device = torch.device("cpu")
    torch.manual_seed(6)
    n_state = 3
    n_ctrl = 3
    n_all = n_state + n_ctrl
    alpha = 0.2
    T = 5
    
    C = torch.squeeze(torch.randn(T, 1, n_all, n_all))
    C = torch.matmul(C.mT, C)
    c = torch.squeeze(torch.randn(T, 1, n_all))
    
    A = torch.eye(n_state) + alpha*torch.randn(n_state, n_state)
    B = torch.randn(n_state, n_ctrl)
    x0 = torch.randn(n_state)
    solver = LQR_Solver()
    posnet = PoseNet(T, C, c, n_all).to(device)
    constraints = ConstrainNet(A, B, T, posnet, x0).to(device)
    print('input:', posnet.input)
    print('initial violation:', torch.norm(constraints()))
    print('initial object:', posnet())
    optimizer = ALM(model=posnet, constraints=constraints, penalty_factor=1, learning_rate=1e-5,penalty_safeguard=1e5, num_iter=8000)

    # Optimization loop
    scheduler = StopOnPlateau(optimizer, steps=100, patience=10, decreasing=-1, verbose=True)
    while scheduler.continual():
        loss = optimizer.step(input)
        scheduler.step(loss)
    print('-----------optimized result----------------')
    decimal_places = 4
    print("Lambda:",optimizer.alm_model.lmd)
    print('tau:', posnet.input)
    print('object f(x):\n', np.around(posnet().detach().numpy(), decimals=decimal_places))
    print('final violation:\n', torch.norm(constraints()))
    tau, mu = solver(A, B, C, c, T, x0)
    print('Lambda:', mu)
    print('tau:\n', tau)
    
    
if __name__ == "__main__":
    test_optim_liegroup()