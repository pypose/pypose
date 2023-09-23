import torch
from torch import nn
from .scndopt import _Optimizer

### Inner Class: updating largragian related parameters
	### Updating largragian related parameters


class _Unconstrained_Model(nn.Module):
    def __init__(self, model, penalty_factor):
        super().__init__()
        self.model = model
        self.pf = penalty_factor

    def update_lambda(self, error):
        self.lmd += error * self.pf
        return self.lmd

    def update_penalty_factor(self, pnf_update_step, safe_guard):
        pf = self.pf * pnf_update_step
        self.pf = pf if pf < safe_guard else safe_guard

    def forward(self, inputs=None, target=None):


        R, C = self.model(inputs)
        self.lmd = self.lmd if hasattr(self, 'lmd') \
                else torch.zeros((C.shape[0], ))
        self.lmd = self.lmd.to(R.device)

        penalty_term = torch.square(torch.norm(C))
        L = R + (self.lmd @ C) + self.pf * penalty_term / 2
        return L

############
    # Update Needed Parameters:
    #   1. model params: \theta, update with SGD
    #   2. lambda multiplier: \lambda, \lambda_{t+1} = \lambda_{t} + pf * error_C
    #   3. penalty factor(Optional): update_para * penalty factor
class SAL(_Optimizer):
    '''
    Stochastic Augmented Lagrangian method for Constraint Optimization.
    '''
    def __init__(self, model, inner_optimizer, inner_scheduler=None, penalty_factor=1, penalty_safeguard=1e5, \
                       penalty_update_factor=2, object_decrease_tolerance=1e-6, violation_tolerance=1e-6, \
                       decrease_rate=0.9, min=1e-6, max=1e32, inner_iter=400):
        defaults = {**{'min':min, 'max':max}}
        super().__init__(model.parameters(), defaults=defaults)
        #### choose your own optimizer for unconstrained opt.
        ### Shared Augments
        self.model = model

        # self.clip_value = clip_value
        self.inner_iter = inner_iter

        # algorithm implemented
        self.terminate = False
        self.decrease_rate = decrease_rate
        self.pf_rate =penalty_update_factor
        self.pf_safeguard = penalty_safeguard
        self.violation_tolerance = violation_tolerance
        self.object_decrease_tolerance = object_decrease_tolerance
        self.alm_model = _Unconstrained_Model(self.model, penalty_factor=penalty_factor)
        self.optim = inner_optimizer
        self.scheduler = inner_scheduler



    #### f(x) - y = loss_0, f(x) + C(x) - 0 - y
    def step(self, inputs=None):
        inner_cost, constrain = self.model(inputs)
        self.best_violation = self.best_violation if hasattr(self, 'best_violation') \
            else torch.norm(constrain)
        self.last_object_value = inner_cost

        self.last = self.loss = self.loss if hasattr(self, 'loss') \
                                    else self.alm_model(inputs)

        for _ in range(self.inner_iter):
            self.optim.zero_grad()
            self.loss = self.alm_model(inputs)
            self.loss.backward()
            self.optim.step()

        # if self.scheduler:
        #     self.scheduler.step()

        with torch.no_grad():
            object_value, violation = self.model(inputs)
            self.log_generation(alm_model=self.alm_model, violation=violation, \
                                last_object_value=self.last_object_value, object_value=object_value)

            if torch.norm(violation) <= torch.norm(self.best_violation) * self.decrease_rate:

                if torch.norm(self.last_object_value-object_value) <= self.object_decrease_tolerance \
                    and torch.norm(violation) <= self.violation_tolerance:
                    print("found optimal")
                    self.terminate = True
                    return self.loss, self.alm_model.lmd

                self.alm_model.update_lambda(violation)
                self.best_violation = violation

            # if violation is not well satisfied, add further punishment
            else:
                self.alm_model.update_penalty_factor(self.pf_rate, self.pf_safeguard)

        return self.loss, self.alm_model.lmd

    def log_generation(self, alm_model, violation, last_object_value, object_value):
        print('--------------------NEW-ALM-EPOCH-------------------')
        print('current_lambda: ', alm_model.lmd)
        # print('parameters: ', alm_model.model.parameters())
        print('object_loss:', object_value)
        print('absolute violation:', torch.norm(violation))
        print("object_loss_decrease", torch.norm(last_object_value-object_value))
