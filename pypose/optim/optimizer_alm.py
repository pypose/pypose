import torch
from torch import nn
from .optimizer import _Optimizer

class _Unconstrained_Model(nn.Module):
    def __init__(self, model, constraints, penalty_factor, lagrange_multiplier):
        super().__init__()
        self.model = model
        self.constraints = constraints
        self.pf = penalty_factor
        self.lmd = lagrange_multiplier
        
    def update_lambda(self, error):
        self.lmd += error * self.pf
        return self.lmd
        
    def update_penalty_factor(self, pnf_update_step, safe_guard):
        pf = self.pf * pnf_update_step
        self.pf = pf if pf < safe_guard else safe_guard
        
    def forward(self, input=None):
        # self.lmd initialize
        R = self.model(input)
        C = self.constraints()
        penalty_term = torch.square(torch.norm(C))
        L = R + torch.matmul(self.lmd, C) + self.pf * penalty_term / 2
        return L

class AugmentedLagrangian(_Optimizer):
    def __init__(self, model, constraints, penalty_factor=1, penalty_safeguard=1e5, \
                       penalty_update_factor=2, object_decrease_tolerance=1e-6, violation_tolerance=1e-6, \
                           min=1e-6, max=1e32, inner_iter=400):  # learning rate add
        defaults = {'min':min, 'max':max}
        super().__init__(model.parameters(), defaults=defaults)        
        self.penalty_factor = penalty_factor
        self.alm_model = _Unconstrained_Model(model, constraints, penalty_factor=self.penalty_factor, \
            lagrange_multiplier=torch.zeros_like(constraints()))
        self.pf_rate = penalty_update_factor
        self.pf_safeguard = penalty_safeguard
        self.best_violation = torch.norm(self.alm_model.constraints())
        self.decrease_rate = 0.9
        self.object_decrease_tolerance = object_decrease_tolerance
        self.violation_tolerance = violation_tolerance
        self.terminate = False
        self.inner_iter = inner_iter
        self.optim = torch.optim.SGD(self.alm_model.parameters(), lr=1e-2, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=20, gamma=0.5)

    def step(self, input, target=None, weight=None, strategy=None):
        self.last_object_value = self.alm_model.model(input)
        optimizer = self.optim
        self.last = self.loss = self.loss if hasattr(self, 'loss') \
                                else self.alm_model(input)
        # print('-------------------------------------------')
        for idx in range(self.inner_iter):
            optimizer.zero_grad()
            self.loss = self.alm_model(input)
            self.loss.backward() 
            optimizer.step()
            # if idx % 50 == 0:
                # print('loss:', self.loss)
        self.scheduler.step()
        with torch.no_grad():
            violation = self.alm_model.constraints()
            # print('lambda: ', self.alm_model.lmd)
            # print('absolute violation:', torch.norm(violation).tolist())
            # print("object decrease", torch.norm(self.last_object_value-self.alm_model.model(input)).item())
            if torch.norm(violation) <= torch.norm(self.best_violation) * self.decrease_rate:
                if torch.norm(self.last_object_value-self.alm_model.model(input)) <= self.object_decrease_tolerance \
                    and torch.norm(violation) <= self.violation_tolerance:
                    print("found optimal")
                    self.terminate = True
                    return self.loss, self.alm_model.lmd, self.terminate
                self.alm_model.update_lambda(violation)
                # print('update lambda')
                self.best_violation = violation
            else:
                self.alm_model.update_penalty_factor(self.pf_rate, self.pf_safeguard)
                # print('update pf: ', self.alm_model.pf)
                
        return self.loss, self.alm_model.lmd, self.terminate