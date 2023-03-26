import copy
import torch
import pypose as pp
from torch import nn
import functorch
from torch.autograd.functional import jacobian
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_functional(mod, disable_autograd_tracking=False):
    # relies on the code snippet from 
    # https://pytorch.org/docs/stable/func.migrating.html?highlight=functorch#functorch-make-functional
    params_dict = dict(mod.named_parameters())
    params_names = params_dict.keys()
    params_values = tuple(params_dict.values())
    
    stateless_mod = copy.deepcopy(mod)
    stateless_mod.to('meta')

    def fmodel(new_params_values, *args, **kwargs):
        new_params_dict = {name: value for name, value in zip(params_names, new_params_values)}
        return torch.func.functional_call(stateless_mod, new_params_dict, args, kwargs)
  
    if disable_autograd_tracking:
        params_values = torch.utils._pytree.tree_map(torch.Tensor.detach, params_values)
    return fmodel, params_values

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
        func, params = make_functional(model)
        J = jacobian(lambda *param: func(param, input), params)
        self.verify_jacobian(J)

        model = nn.Conv2d(2, 2, 2)
        input = torch.randn(2, 2, 2)
        func, params = make_functional(model)
        J = jacobian(lambda *param: func(param, input), params)
        self.verify_jacobian(J)


    def test_pypose_jacobian(self):

        class PoseTransform(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = pp.Parameter(pp.randn_so3(2))
                self.w = pp.Parameter(pp.randn_so3())

            def forward(self):
                return self.p.Exp().tensor()

        model, input = PoseTransform().to(device), pp.randn_SO3(device=device)
        J1 = pp.optim.functional.modjac(model, input=None, flatten=True)
        self.verify_jacobian(J1)

        func, params = make_functional(model)
        jacrev = torch.func.jacrev(lambda *param: func(param, input), params)

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
