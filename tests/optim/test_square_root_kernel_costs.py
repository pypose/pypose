from decimal import Decimal, localcontext

import pytest
import torch
import pypose as pp


KERNELS = [pp.optim.kernel.PseudoHuber, pp.optim.kernel.SoftLOne]


def reference_cost(values, kernel, delta):
    with localcontext() as context:
        context.prec = 80
        scale = Decimal.from_float(delta) ** 2
        result = []
        for value in values.reshape(-1).tolist():
            value = Decimal.from_float(value)
            if kernel is pp.optim.kernel.PseudoHuber:
                cost = 2 * scale * ((1 + value / scale).sqrt() - 1)
            else:
                cost = 2 * ((1 + scale * value).sqrt() - 1)
            result.append(float(cost))
    return values.new_tensor(result).reshape(values.shape)


@pytest.mark.parametrize('kernel', KERNELS)
@pytest.mark.parametrize('delta', [0.25, 1.0, 4.0])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_small_costs_match_high_precision_reference(kernel, delta, dtype):
    epsilon = torch.finfo(dtype).eps
    scale = delta ** 2 if kernel is pp.optim.kernel.PseudoHuber else delta ** -2
    values = torch.tensor([0.0, epsilon / 8, epsilon / 2, epsilon, 0.1, 1.0], dtype=dtype) * scale
    before = values.clone()
    actual = kernel(delta)(values)
    torch.testing.assert_close(actual, reference_cost(values, kernel, delta),
                               rtol=8 * epsilon, atol=0)
    assert actual[0] == 0
    assert torch.all(actual[1:] > 0)
    torch.testing.assert_close(values, before, rtol=0, atol=0)


@pytest.mark.parametrize('kernel', KERNELS)
@pytest.mark.parametrize('delta', [0.25, 1.0, 4.0])
def test_first_and_second_derivatives_match_analytic_form(kernel, delta):
    values = torch.tensor([0.0, 1e-18, 1e-8, 0.2, 2.0, 1e6],
                          dtype=torch.float64, requires_grad=True)
    first, = torch.autograd.grad(kernel(delta)(values).sum(), values, create_graph=True)
    second, = torch.autograd.grad(first.sum(), values)
    scale = delta ** 2
    if kernel is pp.optim.kernel.PseudoHuber:
        base = 1 + values / scale
        expected_first = base.pow(-0.5)
        expected_second = -0.5 / scale * base.pow(-1.5)
    else:
        base = 1 + values * scale
        expected_first = scale * base.pow(-0.5)
        expected_second = -0.5 * scale ** 2 * base.pow(-1.5)
    torch.testing.assert_close(first, expected_first, rtol=2e-12, atol=0)
    torch.testing.assert_close(second, expected_second, rtol=2e-12, atol=0)


@pytest.mark.parametrize('kernel', KERNELS)
@pytest.mark.parametrize('delta', [1.0, 4.0])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_large_costs_remain_finite(kernel, delta, dtype):
    maximum = torch.finfo(dtype).max
    values = torch.tensor([maximum / 4, maximum / 2, maximum], dtype=dtype)
    actual = kernel(delta)(values)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference_cost(values, kernel, delta),
                               rtol=8 * torch.finfo(dtype).eps, atol=0)


@pytest.mark.parametrize('kernel', KERNELS)
def test_empty_and_batched_cost_shapes(kernel):
    for shape in [(0,), (2, 0, 3), (2, 3, 4)]:
        values = torch.full(shape, 1e-8, dtype=torch.float64)
        result = kernel()(values)
        assert result.shape == shape
        torch.testing.assert_close(result, reference_cost(values, kernel, 1.0),
                                   rtol=2e-14, atol=0)


@pytest.mark.parametrize('kernel', KERNELS)
def test_lm_reports_small_positive_residual_objective(kernel):
    class Residual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.value = torch.nn.Parameter(torch.tensor([1e-4], dtype=torch.float32))

        def forward(self, _):
            return self.value.unsqueeze(0)

    model = Residual()
    optimizer = pp.optim.LM(model, kernel=kernel(),
                            strategy=pp.optim.strategy.Constant(damping=1.0))
    cost = optimizer.step(torch.empty(0))
    expected = reference_cost(model.value.detach().square(), kernel, 1.0).sum()
    assert 0 < model.value.item() < 1e-4
    assert cost > 0
    torch.testing.assert_close(cost, expected, rtol=1e-6, atol=0)
