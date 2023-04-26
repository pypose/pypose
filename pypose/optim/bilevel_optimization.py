#### ADD RELATIVE IMPORTS
from .optimizer import _Optimizer
from .solver import Cholesky
from .functional import modjac
####
import torch
from torch.autograd.functional import jacobian


class InnerModel(torch.nn.Module):
    r'''
    The base class for defining inner optimization.
    '''
    def __init__(self) -> None:
        super().__init__()

        # Define all parameters
        # self.param = torch.nn.parameter(all_params_to_optimize, requires_grad = True)

    def inner_cost(self, tau):
        raise NotImplementedError("User needs to implement function to return inner cost.")

    def constraint(self, tau):
        raise NotImplementedError("User needs to implement function to return constraint error.")

    def forward(self, tau):
        r'''
        Compute and return the inner cost and constraint error.
        '''
        return self.inner_cost(tau), self.constraint(tau)


class OuterModel(torch.nn.Module):
    r'''
    The base class for defining outer optimization.
    '''
    def __init__(self, inner_model: torch.nn.Module) -> None:
        super().__init__()

        self.inner_model = inner_model

    def outer_loss(self, tau_star):
        r'''
        Compute outer loss.
        '''
        raise NotImplementedError("User needs to implement function to return outer loss.")

    def forward(self, tau_star):
        r'''
        Compute and return outer loss.
        '''
        return self.outer_loss(tau_star)

class BLO(_Optimizer):
    r'''
    Optimizer for solving bi-level optimization problem.
    '''
    def __init__(self, outer_model: torch.nn.Module, inner_optimizer: torch.optim,
                 solver = None, ) -> None:
        super().__init__(outer_model.parameters(), defaults = {})

        self.inner_optimizer = inner_optimizer
        # self.scheduler = pp.optim.scheduler.StopOnPlateau(inner_optimizer)

        self.outer_model = outer_model

    ####
    def _dLdTheta(self, input):
        return modjac(self.outer_model.forward, input)

    def _dLdTau(self, input):
        return jacobian(self.outer_model.forward, input)

    def _dJGdTau(self, input):
        return jacobian(self.outer_model.inner_model.forward, input, create_graph = True)

    def _dJGdTauTau(self, input):
        return jacobian(self._dJGdTau, input)

    def _dJGdTauTheta(self, input):
        return jacobian(self._dJGdTau, input, create_graph = True)

    def _dGdTheta(self, input):
        return modjac(self.outer_model.inner_model.forward, input)[1]

    def _H(self, input):
        dJdT, dGdT = jacobian(self._dJdGtau, input, create_graph = True)
        return torch.add(dJdT, dGdT, alpha = self.mu_star.unsqueeze(0).transpose(0, 2))

    def _dHdTheta(self, input):
        return modjac(self._H, input)
    ####

    @torch.no_grad()
    def step(self, input, target = None, weight = None):


        # self.scheduler.optimize()
        self.tau_star, self.mu_star = self.inner_optimizer(self.outer_model.inner_model.A, self.outer_model.inner_model.B, self.outer_model.inner_model.C, \
                                                           self.outer_model.inner_model.c, self.outer_model.inner_model.T, input)

        # COMPUTE GRADIENTS
        # Solve equation (6) from pseudocode
        dJdTT, dGdTT = self._dJGdTauTau(self.tau_star)
        P = torch.matmul(self.mu_star.unsqueeze(-1).unsqueeze(-1).expand(-1, 30, 30), dGdTT).sum(dim = 0)
        P = torch.add(dJdTT, P)
        Q = self._dJGdTau(self.tau_star)
        R = self._dLdTau(self.tau_star)
        top = torch.cat((P, Q.transpose(0, 1)), dim=1)
        bottom = torch.cat((Q, torch.zeros(Q.size(0), Q.size(0))), dim=1)
        LHS = torch.cat((top, bottom), dim=0)
        RHS = torch.zeros(LHS.size(0))
        RHS[:, R.size(0)] = R
        lams = torch.linalg.solve(LHS, -RHS)
        lam_tau = lams[:self.tau_star.size(0)].unsqueeze(0).unsqueeze(0)
        lam_mu = lams[self.tau_star.size(0)].unsqueeze(0).unsqueeze(0)

        # Further compute required jacobians
        dHdTh_times_lam = torch.matmul(lam_tau.transpose(0,2), self._dHdTheta(self.tau_star))
        dGdTh_times_lam = torch.matmul(lam_mu.transpose(0,2), self._dGdTheta(self.tau_star))
        dLdTh = self._dLdTheta(self.tau_star)

        dH_plus_dG = tuple(torch.add(tensor1, tensor2) for (tensor1, tensor2) \
                           in zip(dHdTh_times_lam, dGdTh_times_lam))
        # Calculate parameter gradients
        D = tuple(torch.add(tensor1, tensor2) for (tensor1, tensor2) \
                  in zip(dLdTh, dGdTh_times_lam))

        # STEP THE OPTIMIZER
        self.upadte_parameter(self.outer_model.parameters(), step = D)

        # Calculate and return loss
        self.loss = self.outer_model(self.tau_star, target)
        return self.loss
