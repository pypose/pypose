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
from pypose.optim.alm_optimizer import Augmented_Lagrangian_Algorithm as ALA


### Define Constraits Model, inhereted from objective model to create C(\theta)
### We construct the constraint violation to a shape of (N, ...), 
### where N represents the number of constraints.
class ConstrainNet(nn.Module):
    def __init__(self, objective_net):
        super().__init__()
        self.net = objective_net
        self.x = self.net.pose[..., 0][None, ...]
        self.y = self.net.pose[..., 1][None, ...]
    def forward(self, input):
        constraint_violation_0 = (torch.square(self.x) + torch.square(self.y) - 2)
        return constraint_violation_0

class Tensor_objective_func(nn.Module):
    
    def __init__(self, *dim):
        super().__init__()
        init = th.randn(*dim) 
        self.pose = torch.nn.Parameter(torch.randn(1,2))

    def forward(self, input):
        error = -self.pose.prod() 
        return error

def torch_tensor_test():

    torch.manual_seed(0)# 70
    device = torch.device("cpu")
    input = torch.randn((1, 1), device=device)

    invnet = Tensor_objective_func(1, 2).to(device)
    constraints = ConstrainNet(invnet).to(device)
    

    print('initial x:', invnet.pose)
    print('initial object:', invnet(input))
    print('initial violation:', constraints(input))
    
    strategy = Constant(damping=1e-12)
    optimizer = ALA(model=invnet, constraints=constraints, penalty_factor=1e0, num_iter=500)
    scheduler = StopOnPlateau(optimizer, steps=3, patience=5, decreasing=-1, verbose=False)

    # Optimization loop
    while scheduler.continual():
        loss = optimizer.step(input)
        scheduler.step(loss)


    mu_star = optimizer.alm_model.lmd
    pf_star = optimizer.alm_model.pf
    tau_star = invnet.pose.detach().cpu()
    print(pf_star)
    print(f"optimized_mu: \nvalue: {mu_star}, \n\noptimized_tau: \nvalue: {tau_star}")


if __name__ == "__main__":
    torch_tensor_test()