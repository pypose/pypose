#### ADD RELATIVE IMPORTS 
import pypose as pp
from pypose.optim.solver import Cholesky
from pypose.optim.functional import modjac
####
import torch
from torch.autograd.function import jacobian 


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
        return modjac(self.outer_model.forward, input)
    
    def _dLdTau(self, input):
        return jacobian(self.outer_model.forward, input)

    def _dJGdTau(self, input):
        return jacobian(self.outer_model.inner_model.forward, input, retain_graph = True)
    
    def _dJGdTauTau(self, input):
        return jacobian(self._dJGdTau, input)
    
    def _dJGdTauTheta(self, input):
        return jacobian(self._dJGdTau, input, retain_graph = True)

    def _dGdTheta(self, input):
        return modjac(self.outer_model.inner_model.forward, input)[1]
    
    def _H(self, input):
        dJdT, dGdT = jacobian(self._dJdGtau, input, retain_graph = True)
        return torch.add(dJdT, dGdT, alpha = self.mu_star.unsqueeze(0).transpose(0, 2))

    def _dHdTheta(self, input):
        return modjac(self._H, input)
    ####
    
    @torch.no_grad()
    def step(self, input, target = None, weight = None):
        self.scheduler.optimize()
        self.tau_star, self.mu_star = self.inner_optimizer.tau, self.inner_optimizer.mu

        # COMPUTE GRADIENTS
        # Solve equation (6) from pseudocode
        dJdTT, dGdTT = self._dJdGTauTau(self.tau_star)
        P = torch.add(dJdTT, dGdTT, alpha = self.mu_star.unsqueeze(0).transpose(0, 2))
        Q = self._dJGdTau(self.tau_star)
        R = self._dLdTau(self.tau_star)
        top = torch.cat((P, Q.transpose(0, 1)), dim=1)
        bottom = torch.cat((Q, torch.zeros(Q.size(0), Q.size(0))), dim=1)
        LHS = torch.cat((top, bottom), dim=0)
        RHS = torch.zeros(LHS.size(0))
        RHS[:, R.size(0)] = R
        lams = torch.linalg.solve(LHS, -RHS)
        lam_tau = lams[:self.tau_star.size(0)].unsqueeze(0).unsqueeze(0), 
        lam_mu = lams[self.tau_star.size(0)].unsqueeze(0).unsqueeze(0)

        # Further compute required derivatives
        dHdTh_times_lam = torch.mult(lam_tau.transpose(0,2), self._dHdTheta(self.tau_star))
        dGdTh_times_lam = torch.mult(lam_mu.transpose(0,2), self._dGdTheta(self.tau_star))
        dLdTh = self._dLdTheta(self.tau_star)

        dH_plus_dG = tuple(torch.add(tensor1, tensor2) for (tensor1, tensor2) in zip(dHdTh_times_lam, dGdTh_times_lam))
        D = tuple(torch.add(tensor1, tensor2) for (tensor1, tensor2) in zip(dLdTh, dGdTh_times_lam))


        # SET PARAMETER GRADIENTS
        

        # STEP THE OPTIMIZER

        # Calculate and return loss
        self.loss = self.outer_model(self.tau_star, target)
        return self.loss