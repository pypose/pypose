import copy
import io

import pytest
import torch
from torch import nn

from pypose.optim import GN
from pypose.optim.corrector import FastTriggs
from pypose.optim.kernel import Huber, Cauchy, Scale


class StatefulKernel(nn.Module):
    def __init__(self, coefficient, factor):
        super().__init__()
        self.coefficient = nn.Parameter(torch.tensor([[float(coefficient)]]))
        self.register_buffer("factor", torch.tensor(float(factor)))

    def forward(self, input):
        return (input @ self.coefficient) * self.factor


class ScalarFit(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(0., dtype=torch.float64))

    def forward(self, targets):
        return (self.value - targets).unsqueeze(-1)


def make_kernel(name):
    return {"Huber": Huber(), "Cauchy": Cauchy(), "Scale": Scale(.5)}[name]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("name", ["Huber", "Cauchy"])
def test_deepcopy_owns_independent_kernel(dtype, name):
    kernel = make_kernel(name)
    corrector = FastTriggs(kernel)
    residual = torch.tensor([[2., 3.]], dtype=dtype)
    jacobian = torch.eye(2, dtype=dtype)
    expected = corrector(residual, jacobian)
    copied = copy.deepcopy(corrector)
    kernel.delta2 = .25 ** 2
    if name == "Huber":
        kernel.delta = .25
    actual = copied(residual, jacobian)
    for result, original in zip(actual, expected):
        torch.testing.assert_close(result, original)
    assert not torch.allclose(corrector(residual, jacobian)[0], expected[0])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("name", ["Huber", "Cauchy", "Scale"])
def test_builtin_corrector_serialization(dtype, name):
    original = FastTriggs(make_kernel(name))
    residual = torch.tensor([[.25, .5], [2., 3.]], dtype=dtype)
    jacobian = torch.arange(12, dtype=dtype).reshape(4, 3)
    checkpoint = io.BytesIO()
    torch.save(original, checkpoint)
    checkpoint.seek(0)
    restored = torch.load(checkpoint, weights_only=False)
    for actual, expected in zip(restored(residual, jacobian),
                                original(residual, jacobian)):
        torch.testing.assert_close(actual, expected)


def test_corrector_converts_stateful_kernel_dtype():
    corrector = FastTriggs(StatefulKernel(2., .5)).double()
    residual = torch.tensor([[2.], [3.]], dtype=torch.float64)
    jacobian = torch.eye(2, dtype=torch.float64)
    actual_residual, actual_jacobian = corrector(residual, jacobian)
    torch.testing.assert_close(actual_residual, residual)
    torch.testing.assert_close(actual_jacobian, jacobian)
    assert all(p.dtype == torch.float64 for p in corrector.parameters())
    assert all(b.dtype == torch.float64 for b in corrector.buffers())


def test_state_dict_restores_kernel_parameters_and_buffers():
    original = FastTriggs(StatefulKernel(2., .5))
    restored = FastTriggs(StatefulKernel(9., 8.))
    restored.load_state_dict(original.state_dict())
    residual, jacobian = torch.tensor([[2.], [3.]]), torch.eye(2)
    for actual, expected in zip(restored(residual, jacobian),
                                original(residual, jacobian)):
        torch.testing.assert_close(actual, expected)
    assert set(original.state_dict()) == {"kernel.coefficient", "kernel.factor"}


def test_copied_corrector_preserves_gauss_newton_step():
    kernel = Huber(delta=1.)
    copied = copy.deepcopy(FastTriggs(kernel))
    kernel.delta, kernel.delta2 = .25, .25 ** 2
    model = ScalarFit()
    optimizer = GN(model, kernel=Huber(delta=1.), corrector=copied)
    targets = torch.tensor([0., 0., 0., 10.], dtype=torch.float64)
    optimizer.step(targets)
    # At zero, the three inliers have unit weight and the outlier has weight 1/10.
    expected = torch.tensor(1. / 3.1, dtype=targets.dtype)
    torch.testing.assert_close(model.value, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("name", ["Huber", "Cauchy", "Scale"])
def test_existing_corrections_match_analytic_weights(dtype, name):
    residual = torch.tensor([[.2, .3], [2., 3.]], dtype=dtype)
    jacobian = torch.arange(12, dtype=dtype).reshape(4, 3)
    squared_norm = residual.square().sum(-1, keepdim=True)
    if name == "Huber":
        weight = torch.minimum(torch.ones_like(squared_norm), squared_norm.rsqrt())
    elif name == "Cauchy":
        weight = (1 + squared_norm).reciprocal()
    else:
        weight = torch.full_like(squared_norm, .5)
    scale = weight.sqrt()
    actual_residual, actual_jacobian = FastTriggs(make_kernel(name))(residual, jacobian)
    torch.testing.assert_close(actual_residual, scale * residual)
    torch.testing.assert_close(actual_jacobian,
                               scale.repeat_interleave(2, dim=0) * jacobian)
