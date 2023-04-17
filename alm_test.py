import torch
import pypose
import numpy as np
import pypose as pp
from torch import nn
from pypose.lietensor import randn_SE3
from pypose.optim.solver import Cholesky
from pypose.optim.strategy import Constant
from pypose.optim.scheduler import StopOnPlateau
from pypose.optim.optimizer import LevenbergMarquardt as LM
from pypose.optim.alm_optimizer import Augmented_Lagrangian_Algorithm as ALA


    
### Define Objective Model whose Parameters to be Minimzed 
class InvNet(nn.Module):
    
    def __init__(self, *dim):
        super().__init__()
        init = pp.randn_so3(*dim)
        
        self.pose = pp.Parameter(init)

    def forward(self, input):
        error = (self.pose @ input)
        return error.tensor()


### Define Constraits Model, inhereted from objective model to create C(\theta)
class ConstrainNet(nn.Module):
    
    def __init__(self, objective_net):
        super().__init__()
        
        self.net = objective_net
        
    def forward(self, input):
        constraint_violation = self.net.pose - 20*torch.ones_like(input)
        return constraint_violation
    
    
if __name__ == "__main__":
    device = torch.device("cpu")
    input = pp.randn_so3(2, device=device)
    ### instantiated objective and consntraints
    invnet = InvNet(2, 2).to(device)
    ### Feed the params to constraints to obtain C(\theta)
    constraints = ConstrainNet(invnet).to(device)
    strategy = Constant(damping=1e-4)
    constraint_violation = []
    
    losses = [] 
    constraint_violation = []
    objective_loss = []

    ### Setting ALA optimizer, 
    optimizer = ALA(model=invnet, constraints=constraints, penalty_factor=50, num_iter=20)
    scheduler = StopOnPlateau(optimizer, steps=20, patience=10, decreasing=1e-3, verbose=True)

    # 2nd option, step optimization
    while scheduler.continual():
        loss = optimizer.step(input)
        scheduler.step(loss)
        losses.append(torch.norm(loss).cpu().detach().numpy())
        constraint_violation.append(torch.norm(constraints(input)).cpu().detach().numpy())
        objective_loss.append(torch.norm(invnet(input)).cpu().detach().numpy())
        
    mu_star = optimizer.aga_model.lmd
    tau_star = invnet.pose

    print(f"optimized_mu: \nvalue: {mu_star}, \n\noptimized_tau: \nvalue: {tau_star}")
    