import pypose as pp
from pypose.optim.solver import Cholesky
from pypose.optim.functional import modjac
import torch


class InnerModel(torch.nn.Module):
    r'''
    The base class for defining inner optimization.
    '''
    def __init__(self) -> None:
        super().__init__()
        
        # Define all parameters
        self.param = torch.nn.parameter(all_params_to_optimize, requires_grad = True)
    
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
    def __init__(self, outer_model: torch.nn.Module, inner_optimizer: torch.optim,
                 solver = None, ) -> None:
        super().__init__()

        self.inner_optimizer = inner_optimizer
        self.scheduler = pp.optim.scheduler.StopOnPlateau(inner_optimizer)

        self.outer_model = outer_model


    ####
    def _dLdTheta(self, input):
        return modjac(outer_model, input)

    def _dLdTauTheta(self, input):
        return modjac(self._dGdTheta, input)

    def 
    ####
    
    @torch.no_grad()
    def step(self, input, target = None, weight = None):
        self.scheduler.optimize()
        tau_star, mu_star = self.inner_optimizer.tau, self.inner_optimizer.mu
        
        # HOW TO GET JACOBIANS EFFICIENTLY

        # COMPUTE AND SET GRADIENTS

        # STEP THE OPTIMIZER

        self.loss = self.outer_model.forward(tau_star, target)
        return self.outer_model(self.tau_star)