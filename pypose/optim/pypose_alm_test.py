import pypose as pp
from .optimizer import LevenbergMarquardt as LM
from .optimizer import _Optimizer
from .strategy import Constant
from .scheduler import StopOnPlateau
from .solver import Cholesky
from ..lietensor import randn_SE3

import torch
import numpy as np
#import matplotlib.pyplot as plt
from torch import nn
import torch
import pypose



class Unconstrained_Model(nn.Module):
    def __init__(self, model, constraints, en_pf=None, kernel=None, enable_penalty=False):
        super().__init__()  # Call the __init__() method of the parent class
        self.model = model
        self.constraints = constraints
        self.pf = en_pf if en_pf else 0 
        
    def update_lambda(self, error):
    
        self.lmd += error * self.pf
        return self.lmd
        
    def update_penalty_factor(self, pnf_update_step):
        self.pf += pnf_update_step
            
    def forward(self, input, target=None):
        R = self.model(input)
        

        C = self.constraints(input)
        self.lmd = self.lmd if hasattr(self, 'lmd') \
                                    else torch.zeros_like(C) 
        return R + self.lmd * C + self.pf * torch.norm(C) * torch.norm(C)


############
    # Update Needed Parameters:
    #   1. model params: \thetta, gradient_descent, J -> self.update_parameter(pg['params'], D) > D: used linear solver, update with LM
    #   2. lambda multiplier: \lambda, \lambda_{t+1} = \lambda_{t} + pf * error_C -> self.update_parameter(pg['params'], D)
    #   3. penalty factor(Optional): update_para * penalty factor
    
    
class Augmented_Lagrangian_Algorithm(_Optimizer):
    def __init__(self, model, constraints, unconstrained_optimizer=None, enable_penalty=False, penalty_safeguard=1e3, \
                       convergence_tolerance_constrains=1e-3, convergence_tolerance_model= 1e-3, min=1e-6, max=1e32,\
                       update_tolerance=2, scheduler=None, penalty_factor=5, lm_rate=0.01, num_iter=10, kernel=None \
                ):        
        defaults = {**{'min':min, 'max':max}}
        super().__init__(model.parameters(), defaults=defaults)
        #### choose your own optimizer for unconstrained opt.
        self.unconstrained_optimizer =  unconstrained_optimizer if unconstrained_optimizer else LevenbergMarquardt
        
        self.model = model
        self.constraints = constraints #RobustModel(constraints, kernel, auto=False)
        self.lm_rate = lm_rate
        self.num_iter = num_iter
        self.enable_penalty = enable_penalty
        self.convergence_tolerance_constrains = convergence_tolerance_constrains
        self.convergence_tolerance_model = convergence_tolerance_model
        self.update_tolerance = update_tolerance
        self.penalty_safeguard = penalty_safeguard
        self.penalty_factor = penalty_factor
        self.aga_model = Unconstrained_Model(self.model, self.constraints, en_pf=penalty_factor) 
    
    #### f(x) - y = loss_0, f(x) + C(x) - 0 - y 
    @torch.no_grad()
    def step(self, input, target=None, weight=None, strategy=None):
            # convert to unconstraine problem
            optim = self.unconstrained_optimizer(self.aga_model, strategy=strategy)
            self.last = self.loss = self.loss if hasattr(self, 'loss') \
                                    else optim.step(input)       
            #  e.g. LM, it calculates the J of [f(x) - y] to update \theta
            #  for AGA, it calculates the J of [f(x) - y + C - 0 + C_pf] to update \thera, model becomes f(x) + C + C_pf
            # update model parameter \theta
            for _ in range(self.num_iter):
                self.loss = optim.step(input)
            #equality_slack = SE3(torch.zeros_like(input))
            model_eror, error= self.model(input), self.constraints(input)
            print(torch.norm(error), torch.norm(model_eror))
            self.aga_model.update_lambda(error)
            if torch.norm(error) < self.convergence_tolerance_constrains: 
                pnf_update_step = min(self.penalty_safeguard, self.penalty_factor*self.update_tolerance)
                self.aga_model.update_penalty_factor(pnf_update_step)
                
                
            return self.loss
        
           
class Generalized_SGD(_Optimizer):
    def __init__(self, model, solver=None, strategy=None, kernel=None, corrector=None, \
                       learning_rate=0.1, momentum=0.9, min=1e-6, max=1e32, vectorize=True):
        assert min > 0, ValueError("min value has to be positive: {}".format(min))
        assert max > 0, ValueError("max value has to be positive: {}".format(max))
        self.strategy = TrustRegion() if strategy is None else strategy
        defaults = {**{'min':min, 'max':max}, **self.strategy.defaults}
        super().__init__(model.parameters(), defaults=defaults) 
        self.momentum = momentum
        self.lr = learning_rate
        self.model = RobustModel(model, kernel, auto=False)

    @torch.enable_grad()
    def step(self, input, target=None, weight=None):        
        for pg in self.param_groups:
            self.last = self.loss = self.loss if hasattr(self, 'loss') \
                                    else self.model.loss(input, target)
            grad = torch.autograd.grad(self.loss, pg['params'])[0]
            D = -self.lr * grad.reshape(-1,1)
            self.last = self.loss if hasattr(self, 'loss') \
                        else self.model.loss(input, target)
            with torch.no_grad():
                self.update_parameter(params=pg['params'], step=D)
            self.loss = self.model.loss(input, target)
        return self.loss
        
        
def draw_training_graph(losses, constraint_violation, objective_loss):
    # Define the figure and subplots
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    # Plot the first line on the first subplot
    axs[0].plot(losses, '*-')
    axs[0].set_title('Total loss')

    # Plot the second line on the second subplot
    axs[1].plot(constraint_violation, '*-')
    axs[1].set_title('Constraint violation')

    # Plot the third line on the third subplot
    axs[2].plot(objective_loss, '*-')
    axs[2].set_title('Objective loss')
    # Add horizontal line at y=10 on third subplot
    axs[2].axhline(y=torch.norm(input * torch.ones_like(tau_star).to(input.device) * 20).detach().cpu(), linestyle='--', color='r', label='groundtruth')
    # Add legend to third subplot
    axs[2].legend()

    # Set common y-axis label and show the figure
    fig.suptitle('Training Losses')
    plt.show()
    
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
    device = torch.device("cuda:6")
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
    
    print(f'Error: \ngt: {torch.norm(input * torch.ones_like(tau_star).to(input.device) * 20)} - optimized: {torch.norm(invnet(input))}')
    #draw_training_graph(losses, constraint_violation, objective_loss)