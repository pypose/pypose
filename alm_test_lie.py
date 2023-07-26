import torch
import pypose
import math
import torch as th
import numpy as np
import pypose as pp
from torch import nn
import matplotlib.pyplot as plt
from pypose.lietensor import randn_SE3
from pypose.optim.solver import Cholesky
from pypose.optim.strategy import Constant
from pypose.optim.scheduler import StopOnPlateau
from pypose.optim.optimizer import LevenbergMarquardt as LM
from pypose.optim.alm_optimizer import Augmented_Lagrangian_Algorithm as ALM


class ConstrainNet(nn.Module):
    def __init__(self, objective_net):
        super().__init__()
        self.net = objective_net
        
    def forward(self, input=None):
        fixed_euler_angles = np.array([[0.0, 0.0, 0.0]])
        fixed_quaternion = pp.euler2SO3(fixed_euler_angles)
        quaternion = self.net.pose.Exp()
        difference_quaternions = torch.sub(quaternion, fixed_quaternion)
        distance = torch.norm(difference_quaternions, p=2, dim=1).mean()
        d_fixed = 0.35
        constraint_violation = (distance - d_fixed).view(-1)
        
        return constraint_violation

class PoseNet(nn.Module):
    def __init__(self, *dim):
        super().__init__()
        self.pose = pp.Parameter(pp.randn_so3(*dim))

    def forward(self, input):
        input = pp.SO3(input.sum(dim=0, keepdim=True))
        error = (self.pose.Exp() @ input).Log()
        error = error.sum(dim=0, keepdim=True)
        return error

        
class NormPoseNet(nn.Module):
    def __init__(self, objective_net):
        super().__init__()
        self.pose = objective_net.pose

    def forward(self, input):
        return torch.norm((self.pose.Exp() @ input).matrix() - torch.eye(3))

    
    
def test_optim_liegroup():
    torch.manual_seed(77)
    device = torch.device("cpu")
    euler_angles = np.array([[0.0, 0.0, np.pi/4]])  # [roll, pitch, yaw]
    quaternion = pp.euler2SO3(euler_angles)
    quaternion = torch.tensor(quaternion, dtype=torch.float)
    inputs = pp.SO3(quaternion).tensor()
    posnet = PoseNet(1).to(device)
    norm_posnet = NormPoseNet(posnet).to(device)
    initial_pose = posnet.pose.detach().clone()
    decimal_places = 5
    constraints = ConstrainNet(posnet).to(device)

    optimizer = ALM(model=posnet, constraints=constraints, penalty_factor=1, num_iter=800)
    
    scheduler = StopOnPlateau(optimizer, steps=10, patience=10, decreasing=-1, verbose=True)

    # Optimization loop
    while scheduler.continual():
        loss = optimizer.step(inputs)
        scheduler.step(loss)

    mu_star = optimizer.alm_model.lmd
    pf_star = optimizer.alm_model.pf
    tau_star = posnet.pose.detach().cpu()

    
    print('Objective', inputs)
    print(f"\n optimized_mu: \nvalue: {tau_star}" ) 


    
if __name__ == "__main__":
    test_optim_liegroup()