import torch
from torch import nn
from .optimizer import _Optimizer

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
        
    def forward(self, input=None, target=None):

        self.lmd = self.lmd if hasattr(self, 'lmd') \
                else torch.zeros((self.model.constrain(input).shape[0], ))
        R = self.model.inner_cost(input)

        C = self.model.constrain(input)
        penalty_term = torch.square(torch.norm(C))
        L = R + (self.lmd @ C) + self.pf * penalty_term / 2
        return L

############
    # Update Needed Parameters:
    #   1. model params: \thetta, update with SGD
    #   2. lambda multiplier: \lambda, \lambda_{t+1} = \lambda_{t} + pf * error_C 
    #   3. penalty factor(Optional): update_para * penalty factor
class AugmentedLagrangianMethod(_Optimizer):
    def __init__(self, model, penalty_factor=1, penalty_safeguard=1e5, \
                       penalty_update_factor=2, object_decrease_tolerance=1e-6, violation_tolerance=1e-6, momentum=0.9, \
                       decrease_rate=0.9, min=1e-6, max=1e32, inner_scheduler=None, inner_iter=400, learning_rate=1e-2):      
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
        self.optim = torch.optim.SGD(self.alm_model.parameters(), lr=learning_rate, momentum=momentum)
        self.scheduler = inner_scheduler
        if inner_scheduler:
            self.scheduler.optimizer = self.optim
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=scl_step_size, gamma=scl_gamma)


    #### f(x) - y = loss_0, f(x) + C(x) - 0 - y 
    def step(self, input=None):     
        self.best_violation = self.best_violation if hasattr(self, 'best_violation') \
        else torch.norm(self.model.constrain(input=input))

        self.last = self.loss = self.loss if hasattr(self, 'loss') \
                                    else self.alm_model(input)  
        self.last_object_value = self.model.inner_cost(input)
        for _ in range(self.inner_iter):
            self.optim.zero_grad()
            self.loss = self.alm_model(input)
            self.loss.backward() 
            self.optim.step()
        if self.scheduler:
            self.scheduler.step()
            
        with torch.no_grad():
            violation = self.model.constrain(input)
            self.log_generation(alm_model=self.alm_model, violation=violation,\
             inputs=input, last_object_value=self.last_object_value)
            
            if torch.norm(violation) <= torch.norm(self.best_violation) * self.decrease_rate:

                if torch.norm(self.last_object_value-self.alm_model.model.inner_cost(input=input)) <= self.object_decrease_tolerance \
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

    def log_generation(self, alm_model, violation, inputs, last_object_value):
        print('--------------------NEW-ALM-EPOCH-------------------')
        print('current_lambda: ', alm_model.lmd)
        print('parameters: ', alm_model.model.parameters())
        print('object_loss:', self.alm_model.model.inner_cost(inputs))
        print('absolute violation:', torch.norm(violation))
        print("object_loss_decrease", torch.norm(last_object_value-alm_model.model.inner_cost(inputs)))