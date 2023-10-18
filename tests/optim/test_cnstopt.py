import torch
import numpy as np
import pypose as pp
from torch import nn
import numpy as np
from pypose.optim import SAL
from pypose.optim.scheduler import CnstOptSchduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TestOptim:

    def test_tensor(slef):
        class TensorModel(nn.Module):
            def __init__(self, *dim) -> None:
                super().__init__()
                init = torch.randn(*dim)
                self.pose = torch.nn.Parameter(init)

            def obj(self, inputs):
                result = -self.pose.prod()
                return result

            def cnst(self, inputs):
                violation = torch.square(torch.norm(self.pose, p=2)) - 2
                return violation.unsqueeze(0)

            def forward(self, inputs):
                return self.obj(inputs), self.cnst(inputs)

        input = None
        TensorNet = TensorModel(5).to(device)
        # inner_opt = torch.optim.Adam(TensorNet.parameters(), lr=1e-2)
        inner_opt = torch.optim.SGD(TensorNet.parameters(), lr=1e-2, momentum=0.9)
        inner_schd = torch.optim.lr_scheduler.StepLR(optimizer=inner_opt, step_size=20, \
                                                     gamma=0.5, verbose=False)
        optimizer = SAL(model=TensorNet, inner_optimizer=inner_opt, penalty_safeguard=1e3)
        scheduler = CnstOptSchduler(optimizer, steps=30, inner_scheduler=inner_schd, inner_iter=400, \
                                    object_decrease_tolerance=1e-6, violation_tolerance=1e-6, \
                                    verbose=True)

        # scheduler 
        while scheduler.continual():
            loss = optimizer.step(input)
            scheduler.step(loss)
        
        # scheduler
        # scheduler.optimize(input=input)

        print("Lambda*:", optimizer.lagrangeMultiplier)
        print("x*:", TensorNet.pose)

if __name__ == "__main__":
    test = TestOptim()
    test.test_tensor()
