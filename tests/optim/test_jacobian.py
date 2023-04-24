import copy
import torch
import pypose as pp
from torch import nn
from torch.func import functional_call
from torch.utils._pytree import tree_map
from torch.autograd.functional import jacobian
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestJacobian:

    def make_functional(self, mod, disable_autograd_tracking=False):
        # relies on the code snippet from https://tinyurl.com/hrv99cuk
        params_dict = dict(mod.named_parameters())
        params_names = params_dict.keys()
        params_values = tuple(params_dict.values())
        
        stateless_mod = copy.deepcopy(mod)
        stateless_mod.to('meta')

        def fmodel(new_params_values, *args, **kwargs):
            new_params_dict = dict(zip(params_names, new_params_values))
            return functional_call(stateless_mod, new_params_dict, args, kwargs)

        if disable_autograd_tracking:
            params_values = tree_map(torch.Tensor.detach, params_values)
        return fmodel, params_values

    def verify_jacobian(self, J1, J2):
        for j1, j2 in zip(J1, J2):
            assert not torch.any(torch.isnan(j1))
            torch.testing.assert_close(j1, j2)

    def test_tensor_jacobian_single_param(self):

        model = nn.Conv2d(2, 2, 2)
        input = torch.randn(2, 2, 2)
        func, params = self.make_functional(model)
        J1 = jacobian(lambda *param: func(param, input), params)
        J2 = pp.optim.functional.modjac(model, input)
        J3 = pp.optim.functional.modjacrev(model, input)
        J4 = pp.optim.functional.modjacfwd(model, input)
        self.verify_jacobian(J1, J2)
        self.verify_jacobian(J1, J3.values())
        self.verify_jacobian(J1, J4.values())

    def test_tensor_jacobian_multi_param(self):

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = nn.Parameter(torch.randn(2,2,2))
                self.w = nn.Parameter(torch.randn(2,2,2))

            def forward(self, x):
                return self.p * x

        model = Model()
        input = torch.randn(2, 2, 2)
        func, params = self.make_functional(model)
        J1 = jacobian(lambda *param: func(param, input), params)
        J2 = pp.optim.functional.modjac(model, input)
        self.verify_jacobian(J1, J2)

    def test_lietensor_jacobian(self):

        class PoseTransform(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = pp.Parameter(pp.randn_so3(2))
                self.w = pp.Parameter(pp.randn_so3())

            def forward(self):
                return self.p.Exp().tensor()

        model = PoseTransform().to(device)

        func, params = self.make_functional(model)
        J1 = jacobian(lambda *param: func(param), params)
        J2 = pp.optim.functional.modjac(model, input=None, flatten=False)
        self.verify_jacobian(J1, J2)

        model = PoseTransform().to(device)
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
        assert not torch.any(torch.isnan(J))


if __name__ == '__main__':
    test = TestJacobian()
    test.test_tensor_jacobian_single_param()
    test.test_tensor_jacobian_multi_param()
    test.test_lietensor_jacobian()
