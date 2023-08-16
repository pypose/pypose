import torch
import numpy as np
import pypose as pp
from torch import nn
import numpy as np
from pypose.optim.optimizer_alm import AugmentedLagrangianMethod as ALM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TestOptim:

    def test_tensor(slef):
        class TensorModel(nn.Module):
            def __init__(self, *dim) -> None:
                super().__init__()
                init = torch.randn(*dim)
                self.pose = torch.nn.Parameter(init)
                
            def objective(self, inputs):
                result = -self.pose.prod()
                return result

            def constrain(self, inputs):
                constraint_violation = torch.square(torch.norm(self.pose, p=2)) - 2
                return constraint_violation.unsqueeze(0)

            def forward(self, inputs):
                return self.objective(inputs), self.constrain(inputs)
        input = None
        TensorNet = TensorModel(5).to(device)

        inner_optimizer = torch.optim.SGD(TensorNet.parameters(), lr=1e-2, momentum=0.9)
        inner_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=inner_optimizer, step_size=20, gamma=0.5)
        # scheduler = pp.optim.scheduler.StopOnPlateau(optimizer, steps=10, patience=3, decreasing=1e-3, verbose=True)
        optimizer = ALM(model=TensorNet, inner_optimizer=inner_optimizer, inner_scheduler=inner_scheduler, inner_iter=400, penalty_safeguard=1e3)
        
        for idx in range(20):
            loss, lmd, = optimizer.step(input)
            if optimizer.terminate:
                break
        print('-----------optimized result----------------')
        print("Lambda*:", lmd)
        print("x*:", TensorNet.pose)
    
if __name__ == "__main__":
    test = TestOptim()
    test.test_tensor()