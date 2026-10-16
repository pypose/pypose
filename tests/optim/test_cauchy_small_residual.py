import math

import pytest
import torch
import pypose as pp



@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('delta', [0.1, 1.0, 10.0])
def test_nonzero_costs_below_machine_epsilon(dtype, delta):
    scale = delta * delta
    x = torch.tensor([0.0, 1e-18, 1e-12, 1e-9, 1e-7, 0.1, 1.0], dtype=dtype) * scale
    expected = torch.tensor([scale * math.log1p(float(value) / scale) for value in x], dtype=dtype)
    actual = pp.optim.kernel.Cauchy(delta)(x)
    torch.testing.assert_close(actual, expected, rtol=5e-7 if dtype == torch.float32 else 2e-14, atol=0)
    assert torch.all(actual[1:] > 0)


@pytest.mark.parametrize('delta', [0.1, 1.0, 10.0])
def test_first_and_second_derivatives_match_robust_cost(delta):
    x = torch.tensor([0.0, 1e-18, 1e-8, 0.2, 2.0], dtype=torch.float64, requires_grad=True)
    cost = pp.optim.kernel.Cauchy(delta)(x)
    first, = torch.autograd.grad(cost.sum(), x, create_graph=True)
    second, = torch.autograd.grad(first.sum(), x)
    scale = delta * delta
    torch.testing.assert_close(first, scale / (scale + x))
    torch.testing.assert_close(second, -scale / (scale + x).square())


def test_large_costs_and_parameter_gradient():
    x = torch.tensor([1e100, 1e200, 1e300], dtype=torch.float64)
    actual = pp.optim.kernel.Cauchy()(x)
    torch.testing.assert_close(actual, torch.tensor([math.log1p(float(value)) for value in x], dtype=torch.float64))
    parameter = torch.nn.Parameter(torch.tensor([1e-5, 2e-5], dtype=torch.float32))
    cost = pp.optim.kernel.Cauchy()(parameter.square()).sum()
    assert cost.item() > 0
    cost.backward()
    torch.testing.assert_close(parameter.grad, 2 * parameter.detach() / (1 + parameter.detach().square()))
