import math

import pytest
import torch
import pypose as pp



def scalar_tolerant(x, a, b):
    z, z0 = (x - a) / b, -a / b
    softplus = lambda value: max(value, 0.0) + math.log1p(math.exp(-abs(value)))
    return b * (softplus(z) - softplus(z0))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('a,b', [(1.0, -1.0), (1000.0, -1.0), (100.0, -0.01)])
def test_finite_cost_when_transition_exponent_overflows(dtype, a, b):
    x = torch.tensor([0.0, 0.25 * a, 0.75 * a, a, 1.25 * a, 2 * a], dtype=dtype)
    expected = torch.tensor([scalar_tolerant(float(value), a, b) for value in x], dtype=dtype)
    actual = pp.optim.kernel.Tolerant(a=a, b=b)(x)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-5 if dtype == torch.float32 else 2e-12)
    assert actual[0] == 0


def test_stable_derivatives_through_large_transition():
    a, b = 1000.0, -1.0
    x = torch.tensor([0.0, 999.5, 1000.0, 1000.5, 2000.0], dtype=torch.float64, requires_grad=True)
    value = pp.optim.kernel.Tolerant(a=a, b=b)(x)
    first, = torch.autograd.grad(value.sum(), x, create_graph=True)
    second, = torch.autograd.grad(first.sum(), x)
    sigmoid = torch.sigmoid((x - a) / b)
    torch.testing.assert_close(first, sigmoid)
    torch.testing.assert_close(second, sigmoid * (1 - sigmoid) / b)


def test_regular_parameters_match_literal_formula():
    x = torch.tensor([0.0, 0.25, 0.75, 1.0, 2.0, 3.0], dtype=torch.float64)
    a, b = 1.3, -0.7
    expected = b * torch.log1p(torch.exp((x - a) / b)) - b * math.log1p(math.exp(-a / b))
    torch.testing.assert_close(pp.optim.kernel.Tolerant(a=a, b=b)(x), expected, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize('dtype', [torch.int32, torch.int64])
def test_integer_inputs_preserve_fractional_parameters_and_promotion(dtype):
    a, b = 1.3, -0.7
    x = torch.tensor([0, 1, 2, 3], dtype=dtype)
    expected = torch.tensor([scalar_tolerant(float(value), a, b) for value in x])
    actual = pp.optim.kernel.Tolerant(a=a, b=b)(x)
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected)
    assert actual[0] == 0
