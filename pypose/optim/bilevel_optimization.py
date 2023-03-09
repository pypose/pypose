import pypose as pp
import torch

class InnerModel(torch.nn.Module):
    r'''
    The base class for defining inner optimization.
    '''
    def __init__(self) -> None:
        super().__init__()
        
        # Define all parameters
        self.param = torch.nn.parameter(some_param, requires_grad = True)
    
    def forward(tau):
        r'''
        Compute the inner cost and constraint error.
        '''
        raise NotImplementedError("User needs to implement function to return inner cost and constraint error.")
    
class OuterModel(torch.nn.Module):
    r'''
    The base class for defining outer optimization.
    '''
    def __init__(self, inner_model: torch.nn.Module) -> None:
        super().__init__()

        self.inner_model = inner_model

    def forward(tau_star):
        r'''
        Computer outer loss.
        '''
        raise NotImplementedError("User needs to implement function to return outer loss.")

class BLO(pp.optim._Optimizer):
    r'''
    Optimizer for solving bi-level optimization problem.
    '''
    def __init__(self, inner_optimizer) -> None:
        super().__init__()

        self.inner_optimizer = inner_optimizer
        self.scheduler = pp.optim.scheduler.StopOnPlateau(inner_optimizer)

        self.tau_star = None
        self.mu_star = None
    
    def step(self):
        self.scheduler.optimizer()
        self.tau_star, self.mu_star = self.inner_optimizer.tau, self.inner_optimizer.mu
