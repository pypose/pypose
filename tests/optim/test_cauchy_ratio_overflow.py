from decimal import Decimal, localcontext

import pytest
import torch
import pypose as pp


@pytest.mark.parametrize('dtype,large', [
    (torch.float16, 64.), (torch.bfloat16, 1e34),
    (torch.float32, 1e34), (torch.float64, 1e308),
])
def test_cauchy_finite_large_ratio(dtype, large):
    delta = .001
    x = torch.tensor([0., delta**2, large], dtype=dtype, requires_grad=True)
    with localcontext() as context:
        context.prec = 80
        scale = Decimal(delta**2)
        values = [Decimal(float(value)) for value in x.detach()]
        costs = [float(scale * (1 + value / scale).ln()) for value in values]
        derivatives = [float(scale / (scale + value)) for value in values]
    actual = pp.optim.kernel.Cauchy(delta)(x)
    gradient, = torch.autograd.grad(actual.sum(), x)
    tolerance = 4 * torch.finfo(dtype).eps

    assert torch.isfinite(actual).all()
    assert torch.isfinite(gradient).all()
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, torch.tensor(costs, dtype=dtype),
                               rtol=tolerance, atol=0)
    torch.testing.assert_close(gradient, torch.tensor(derivatives, dtype=dtype),
                               rtol=tolerance, atol=torch.finfo(dtype).tiny * torch.finfo(dtype).eps)


def test_cauchy_derivatives_at_scale_boundary():
    x = torch.tensor([.5, 1., 2.], dtype=torch.float64, requires_grad=True)
    kernel = pp.optim.kernel.Cauchy()
    assert torch.autograd.gradcheck(kernel, (x,))
    assert torch.autograd.gradgradcheck(kernel, (x,))


def test_cauchy_retains_integer_input_promotion():
    x = torch.tensor([0, 1, 2, 9])
    actual = pp.optim.kernel.Cauchy(delta=.5)(x)
    assert actual.dtype == torch.get_default_dtype()
    torch.testing.assert_close(actual, .25 * torch.log1p(x / .25))


def test_cauchy_retains_tensor_scale_gradient():
    delta = torch.tensor(.2, dtype=torch.float64, requires_grad=True)
    x = torch.tensor([0., .04, .5, 2.], dtype=delta.dtype)
    cost = pp.optim.kernel.Cauchy(delta)(x)
    gradient, = torch.autograd.grad(cost.sum(), delta)
    expected = 2 * delta * (torch.log1p(x / delta.square())
                            - x / (delta.square() + x)).sum()
    torch.testing.assert_close(gradient, expected)


@pytest.mark.parametrize('dtype,delta', [(torch.float32, 1e-20), (torch.float64, 1e-160)])
def test_cauchy_retains_subnormal_scale_gradients(dtype, delta):
    x = torch.tensor([0., delta**2, 2 * delta**2], dtype=dtype, requires_grad=True)
    reference = delta**2 * torch.log1p(x / delta**2)
    expected_gradient, = torch.autograd.grad(reference.sum(), x)
    actual = pp.optim.kernel.Cauchy(delta)(x)
    gradient, = torch.autograd.grad(actual.sum(), x)

    assert torch.isfinite(actual).all()
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(gradient, expected_gradient, rtol=0, atol=0)


def test_cauchy_large_ratio_allows_gauss_newton_update():
    class ScaledFit(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.value = torch.nn.Parameter(torch.tensor([.001]))

        def forward(self, coefficient):
            return (coefficient * self.value).reshape(1, 1)

    model = ScaledFit()
    optimizer = pp.optim.GN(model, kernel=pp.optim.kernel.Cauchy(delta=.001))
    coefficient, target = torch.tensor(1e20), torch.tensor([[2e17]])
    loss = optimizer.step(coefficient, target)

    assert torch.isfinite(loss)
    torch.testing.assert_close(model.value, (target / coefficient).reshape(1))
