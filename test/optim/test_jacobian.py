import torch
import pypose as pp
from torch import nn
import functorch, functools
from torch.autograd.functional import jacobian
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestJacobian:

    def verify_jacobian(self, J):
        for j in J:
            assert not torch.any(torch.isnan(j))

    def test_torch_jacobian(self):

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = nn.Parameter(torch.randn(2,2,2))
                self.w = nn.Parameter(torch.randn(2,2,2))

            def forward(self, x):
                return self.p * x

        model = Model()
        input = torch.randn(2, 2, 2)
        func, params = functorch.make_functional(model)
        J = jacobian(lambda *param: func(param, input), params)
        self.verify_jacobian(J)

        model = nn.Conv2d(2, 2, 2)
        input = torch.randn(2, 2, 2)
        func, params = functorch.make_functional(model)
        J = jacobian(lambda *param: func(param, input), params)
        self.verify_jacobian(J)


    def test_pypose_jacobian(self):

        class PoseTransform(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = pp.Parameter(pp.randn_so3(2))
                self.w = pp.Parameter(pp.randn_so3())

            def forward(self):
                return self.p.Exp()

        model, input = PoseTransform().to(device), pp.randn_SO3(device=device)
        J1 = pp.optim.functional.modjac(model, input=None, flatten=True)
        self.verify_jacobian(J1)

        func, params = functorch.make_functional(model)
        jacrev = functorch.jacrev(lambda *param: func(param, input), params)

        model, input = PoseTransform().to(device), pp.randn_SO3(device=device)
        huber = pp.optim.kernel.Huber(delta=0.01)

        class RobustModel(nn.Module):
            def __init__(self, module, kernel):
                super().__init__()
                self.module = module
                self.kernel = kernel

            def forward(self, *args, **kwargs):
                return self.kernel(self.module(*args, **kwargs).abs())

        model = RobustModel(model, huber)
        J = pp.optim.functional.modjac(model, input=None, flatten=True)
        self.verify_jacobian(J)


if __name__ == '__main__':
    test = TestJacobian()
    test.test_torch_jacobian()
    test.test_pypose_jacobian()
