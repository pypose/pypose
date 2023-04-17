
import torch
import pypose
import numpy as np
from torch import nn
from .optimizer import _Optimizer
from .optimizer import LevenbergMarquardt as LM


class _Unconstrained_Model(nn.Module):
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
    #   1. model params: \thetta, gradient_descent, J -> self.update_parameter(pg['params'], D) > 
    #	D: used linear solver, update with LM
    #   2. lambda multiplier: \lambda, \lambda_{t+1} = \lambda_{t} + pf * error_C 
    #	-> self.update_parameter(pg['params'], D)
    #   3. penalty factor(Optional): update_para * penalty factor
class Augmented_Lagrangian_Algorithm(_Optimizer):
    def __init__(self, model, constraints, unconstrained_optimizer=None, enable_penalty=False, penalty_safeguard=1e3, \
                       convergence_tolerance_constrains=100, convergence_tolerance_model= 1e-3, min=1e-6, max=1e32,\
                       update_tolerance=2, scheduler=None, penalty_factor=5, lm_rate=0.01, num_iter=10, kernel=None \
                ):        
        defaults = {**{'min':min, 'max':max}}
        super().__init__(model.parameters(), defaults=defaults)
        #### choose your own optimizer for unconstrained opt.
        self.unconstrained_optimizer =  unconstrained_optimizer if unconstrained_optimizer \
        								else LM
        
        self.model = model
        self.constraints = constraints
        self.lm_rate = lm_rate
        self.num_iter = num_iter
        self.enable_penalty = enable_penalty
        self.convergence_tolerance_constrains = convergence_tolerance_constrains
        self.convergence_tolerance_model = convergence_tolerance_model
        self.update_tolerance = update_tolerance
        self.penalty_safeguard = penalty_safeguard
        self.penalty_factor = penalty_factor
        self.aga_model = _Unconstrained_Model(self.model, self.constraints, en_pf=penalty_factor) 
    
    #### f(x) - y = loss_0, f(x) + C(x) - 0 - y 
    @torch.no_grad()
    def step(self, input, target=None, weight=None, strategy=None):
            # convert to unconstraine problem
            optim = self.unconstrained_optimizer(self.aga_model, strategy=strategy)
            self.last = self.loss = self.loss if hasattr(self, 'loss') \
                                    else optim.step(input)       
            #  e.g. LM, it calculates the J of [f(x) - y] to update \theta
            #  for AGA, it calculates the J of [f(x) - y + C - 0 + C_pf] to update \theta
            # update model parameter \theta
            for _ in range(self.num_iter):
                self.loss = optim.step(input)
            model_eror, error= self.model(input), self.constraints(input)
            print(torch.norm(error), torch.norm(model_eror))
            self.aga_model.update_lambda(error)
            if torch.norm(error) < self.convergence_tolerance_constrains: 
                pnf_update_step = min(self.penalty_safeguard, self.penalty_factor * self.update_tolerance)
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
