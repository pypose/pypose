import pypose as pp
import torch

class someModel(torch.nn.Module):
    # Define model here

class LowOptimization(torch.nn.Module):
    def __init__(self, model, constraints, solver, etc.):
        super.__init__()
        self.parameters = model.parameters

    def func(self):
        def forward(self) -> tuple(tau, mu):
            result = self.solver.solve(self.model, self.constraints)
            self.constrains = result.constraints
            return result.optimized_parameters
        
        def backward():
    

class HighOptimization(torch.nn.Module):
    def __init__(self, low_optimizer, torch.optim.optimizer, etc):
        super.__init__()
        self.parameters = low_optimizer.parameters

    def forward(self):
        tau, mu = low_optimizer.forward()
        # some loss calculation
        return loss
    
    def backward():
    
    def solve():
        while(not_optimized):
            loss=forward
            loss.backward
            optimizer
        return self.low_optimizer.model.parameters()

if __name__ == "__main__":
    model = someModel
    low_opt = LowOptimization(model, etc)
    high_opt = HighOptimization()
    optimized_parameters = high_opt.solve()