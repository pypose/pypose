import torch
from torch import nn
from pypose.optim import SAL
from pypose.utils import Prepare
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pypose.optim.scheduler import StopOnPlateau

class TestOptim:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_stoauglag(self):

        class Objective(nn.Module):
            def __init__(self, *dim) -> None:
                super().__init__()
                init = torch.randn(*dim)
                self.pose = torch.nn.Parameter(init)

            def obj(self):
                result = -self.pose.prod()
                return result

            def cnstr(self):
                violation = torch.square(torch.norm(self.pose, p=2)) - 2
                return violation.unsqueeze(0)

            def forward(self):
                '''
                Has to output a 2-tuple, including the error of objective and constraint.
                '''
                return self.obj(), self.cnstr()

        model = Objective(5).to(self.device)
        inopt = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
        insch = Prepare(ReduceLROnPlateau, inopt, "min")
        outopt = SAL(model, scheduler=insch, steps=400, penalty=1, shield=1e3, scale=2, hedge=0.9)
        outsch = StopOnPlateau(outopt, steps=30, patience=1, decreasing=1e-6, verbose=True)

        while outsch.continual():
            loss = outopt.step()
            outsch.step(loss)

        print("Lambda*:", outopt.auglag.lmd)
        print("x*:", model.pose)

    def test_sag_anyinput(self):

        class Objective(nn.Module):
            def __init__(self, *dim) -> None:
                super().__init__()
                init = torch.randn(*dim)
                self.pose = torch.nn.Parameter(init)

            def obj(self, a):
                result = -self.pose.prod() * a
                return result

            def cnstr(self, p):
                violation = torch.square(torch.norm(self.pose, p=p)) - 2
                return violation.unsqueeze(0)

            def forward(self, a, p=2):
                '''
                Take any input, has to output 2-tuple, including a scalar error and vector
                constraints.
                '''
                return self.obj(a), self.cnstr(p)

        model = Objective(5).to(self.device)
        inopt = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
        insch = Prepare(ReduceLROnPlateau, inopt, "min")
        outopt = SAL(model, scheduler=insch, steps=400, penalty=1, shield=1e3, scale=2, hedge=0.9)
        outsch = StopOnPlateau(outopt, steps=30, patience=1, decreasing=1e-6, verbose=True)

        while outsch.continual():
            loss = outopt.step(2, p=2)
            outsch.step(loss)

        print("Lambda*:", outopt.auglag.lmd)
        print("x*:", model.pose)

if __name__ == "__main__":
    test = TestOptim()
    test.test_stoauglag()
    test.test_sag_anyinput()
