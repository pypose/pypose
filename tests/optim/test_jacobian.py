import copy
import torch
import importlib
import pypose as pp
from torch import nn
from contextlib import contextmanager
from typing import Collection, Callable
from torch.utils._pytree import tree_map
from torch.autograd.functional import jacobian
from torch.func import functional_call, jacfwd, jacrev

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

    def test_modjac(self):

        class PoseInv(nn.Module):
            def __init__(self, *dim):
                super().__init__()
                self.pose = pp.Parameter(pp.randn_SE3(*dim))

            def forward(self, inputs):
                error = (self.pose @ inputs).Log().tensor()
                constraint = self.pose.Log().tensor().sum(-1)
                return error, constraint

        B1, B2, M, N = 2, 3, 2, 2
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = pp.randn_SE3(B2, B1, M, N, sigma=0.0001).to(device)
        invnet = PoseInv(M, N).to(device)
        jackwargs = {'vectorize': True, 'flatten': False}
        J = pp.optim.functional.modjac(invnet, input=inputs, **jackwargs)
        assert not pp.hasnan(J)

    def test_lietensor_jacfwd(self):
        pose = pp.randn_SE3(1).to(device)
        points = torch.randn(1, 3).to(device)

        def func(pose, points):
            return pose @ points

        try: # the torch behavior since ver 2.0.0
            jac_func = jacfwd(func)
            jac = jac_func(pose, points)
            raise AssertionError('should not reach here')
        except RuntimeError as e:
            assert 'shapes cannot be multiplied' in str(e)

    def test_lietensor_jacrev(self):
        pose = pp.randn_SE3(1).to(device)
        points = torch.randn(1, 3).to(device)

        def func(pose, points):
            return pose @ points

        @contextmanager
        def check_fn_equal(TO_BE_CHECKED: Collection[Callable]):
            # assert func1 and func2 are equal according to memory reference, module name,
            # function name, and bytecode
            def assert_fn_equal(func1, func2):
                assert func1 == func2 \
                and func1.__module__  == func2.__module__ \
                and func1.__name__  == func2.__name__ \
                and func1.__code__.co_code == func2.__code__.co_code
            try:
                yield
            finally:
                # make sure functions has been restored
                for func1 in TO_BE_CHECKED:
                    module, name = func1.__module__, func1.__name__
                    module = importlib.import_module(module)
                    func2 = getattr(module, name)
                    assert_fn_equal(func1, func2)

        # save functions to be checked
        TO_BE_CHECKED = {
            torch.autograd.forward_ad.make_dual,
            torch._functorch.eager_transforms._wrap_tensor_for_grad,
        }

        with check_fn_equal(TO_BE_CHECKED):
            with pp.retain_ltype():
                jac_func = jacrev(func)
                jac = jac_func(pose, points)
                assert not pp.hasnan(jac)

        # without context manager, call pp.func.jacrev
        with check_fn_equal(TO_BE_CHECKED):
            jac_func = pp.func.jacrev(func)
            jac = jac_func(pose, points)
            assert not pp.hasnan(jac)

if __name__ == '__main__':
    test = TestJacobian()
    test.test_tensor_jacobian_single_param()
    test.test_tensor_jacobian_multi_param()
    test.test_lietensor_jacobian()
    test.test_modjac()
    test.test_lietensor_jacfwd()
    test.test_lietensor_jacrev()
