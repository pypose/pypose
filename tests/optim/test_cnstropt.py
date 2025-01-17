import torch
from torch import nn
from pypose.optim import SAL
from torch.optim.lr_scheduler import StepLR
from pypose.optim.scheduler import CnstrScheduler


class TestOptim:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_tensor(self):

        class TensorModel(nn.Module):
            def __init__(self, *dim) -> None:
                super().__init__()
                init = torch.randn(*dim)
                self.pose = torch.nn.Parameter(init)

            def obj(self, inputs):
                result = -self.pose.prod()
                return result

            def cnstr(self, inputs):
                violation = torch.square(torch.norm(self.pose, p=2)) - 2
                return violation.unsqueeze(0)

            def forward(self, inputs):
                return self.obj(inputs), self.cnstr(inputs)

        input = None
        TensorNet = TensorModel(5).to(self.device)
        # inner_opt = torch.optim.Adam(TensorNet.parameters(), lr=1e-2)
        inner_opt = torch.optim.SGD(TensorNet.parameters(), lr=1e-2, momentum=0.9)
        inner_schd = StepLR(optimizer=inner_opt, step_size=20, gamma=0.5, verbose=False)
        optimizer = SAL(model=TensorNet, optim=inner_opt, penalty_safeguard=1e3)
        scheduler = CnstrScheduler(optimizer, steps=30, inner_scheduler=inner_schd,
                                    inner_iter=400, object_decrease_tolerance=1e-6,
                                    violation_tolerance=1e-6, verbose=True)

        while scheduler.continual():
            loss = optimizer.step(input)
            scheduler.step(loss)

        print("Lambda*:", optimizer.lagrangeMultiplier)
        print("x*:", TensorNet.pose)


if __name__ == "__main__":
    test = TestOptim()
    test.test_tensor()
