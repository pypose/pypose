import torch
import pypose as pp
from torch import nn
from pypose.optim import SAL
from pypose.utils import Prepare
from pypose.optim.scheduler import StopOnPlateau
from torch.optim.lr_scheduler import ReduceLROnPlateau


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
                violation = torch.square(torch.norm(self.pose, p=2)) - 5
                return violation.unsqueeze(0)

            def forward(self):
                '''
                Has to output a 2-tuple, including the error of objective and constraint.
                '''
                return self.obj(), self.cnstr()

        model = Objective(5).to(self.device)

        outopt = SAL(model, penalty=1, shield=1e3, scale=2, hedge=0.9)
        outsch = StopOnPlateau(outopt, steps=30, patience=1, decreasing=1e-6, verbose=True)

        inopt = torch.optim.SGD(outopt.model.parameters(), lr=1e-2, momentum=0.9)
        outopt.inner_scheduler(Prepare(ReduceLROnPlateau, inopt, "min"), steps=400)

        while outsch.continual():
            loss = outopt.step()
            outsch.step(loss)

        result = model.pose.abs().cpu()
        expect = torch.ones(5)
        pp.testing.assert_close(result, expect, atol=1e-5, rtol=1e-5, msg="Failed!")


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
                violation = torch.square(torch.norm(self.pose, p=p)) - 5
                return violation.unsqueeze(0)

            def forward(self, a, p=2):
                '''
                Take any input, has to output 2-tuple, including a scalar error and vector
                constraints.
                '''
                return self.obj(a), self.cnstr(p)

        model = Objective(5).to(self.device)

        outopt = SAL(model, penalty=1, shield=1e3, scale=2, hedge=0.9)
        outsch = StopOnPlateau(outopt, steps=30, patience=1, decreasing=1e-6, verbose=True)

        inopt = torch.optim.SGD(outopt.model.parameters(), lr=1e-2, momentum=0.9)
        outopt.inner_scheduler(Prepare(ReduceLROnPlateau, inopt, "min"), steps=400)

        while outsch.continual():
            loss = outopt.step(2, p=2)
            outsch.step(loss)

        result = model.pose.abs().cpu()
        expect = torch.ones(5)
        pp.testing.assert_close(result, expect, atol=1e-5, rtol=1e-5, msg="Failed!")


if __name__ == "__main__":
    test = TestOptim()
    test.test_stoauglag()
    test.test_sag_anyinput()
