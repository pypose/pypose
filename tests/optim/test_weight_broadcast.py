import math

import pytest
import torch
import pypose as pp
from torch import nn

from pypose.optim.optimizer import RobustModel


def matrix_weights(batch_shape, size, dtype):
    shape = (*batch_shape, size, size)
    values = torch.arange(math.prod(shape), dtype=dtype).reshape(shape) / 17
    return values @ values.mT + torch.eye(size, dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("size", [1, 2, 3])
@pytest.mark.parametrize("batch_shape", [
    (2, 1, 2), (2, 3, 1), (1, 3, 1), (2, 1, 1),
    (), (2,), (3, 2), (2, 3, 2),
])
def test_weighted_residual_and_jacobian_follow_broadcasting(dtype, size, batch_shape):
    residual = torch.arange(12 * size, dtype=dtype).reshape(2, 3, 2, size) / 7
    weight = matrix_weights(batch_shape, size, dtype)
    jacobian = torch.arange(residual.numel() * 3, dtype=dtype).reshape(-1, 3) / 13
    original = weight.clone()

    flat, block, jac = RobustModel(nn.Identity()).normalize_RWJ(
        [residual], [weight], [jacobian]
    )

    expected_residual = (weight @ residual.unsqueeze(-1)).squeeze(-1).reshape(-1)
    expected_jacobian = (weight @ jacobian.reshape(*residual.shape, 3)).reshape(-1, 3)
    torch.testing.assert_close(block @ flat, expected_residual)
    torch.testing.assert_close(block @ jac, expected_jacobian)
    torch.testing.assert_close(weight, original, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_strided_weight_batches_keep_their_residual_order(dtype):
    weight = matrix_weights((2, 3, 2), 2, dtype)
    weight = weight.transpose(0, 1).contiguous().transpose(0, 1)
    assert not weight.is_contiguous()
    residual = torch.arange(24, dtype=dtype).reshape(2, 3, 2, 2)

    flat, block, _ = RobustModel(nn.Identity()).normalize_RWJ(
        [residual], [weight], [torch.ones(24, 1, dtype=dtype)]
    )

    expected = (weight @ residual.unsqueeze(-1)).squeeze(-1).reshape(-1)
    torch.testing.assert_close(block @ flat, expected)


def test_multiple_residual_groups_use_separate_weight_broadcasts():
    residuals = [torch.arange(24.).reshape(2, 3, 2, 2), torch.arange(6.).reshape(3, 2, 1)]
    weights = [matrix_weights((2, 1, 2), 2, torch.float32),
               matrix_weights((3, 1), 1, torch.float32)]
    jacobians = [torch.ones(r.numel(), 2) for r in residuals]

    flat, block, jac = RobustModel(nn.Identity()).normalize_RWJ(residuals, weights, jacobians)

    expected = torch.cat([(w @ r.unsqueeze(-1)).reshape(-1)
                          for r, w in zip(residuals, weights)])
    torch.testing.assert_close(block @ flat, expected)
    torch.testing.assert_close(jac, torch.cat(jacobians))


@pytest.mark.parametrize("shape", [(2, 3, 2, 1), (2, 3, 1, 1)])
@pytest.mark.parametrize("shared", [False, True])
def test_existing_scalar_weight_shorthand(shape, shared):
    residual = torch.arange(math.prod(shape), dtype=torch.float64).reshape(shape)
    weight = torch.tensor([3.], dtype=torch.float64) if shared else residual + 1

    flat, block, _ = RobustModel(nn.Identity()).normalize_RWJ(
        [residual], [weight], [torch.ones(residual.numel(), 1)]
    )

    torch.testing.assert_close(block @ flat, (weight * residual).reshape(-1))


def test_unweighted_residual_groups_are_unchanged():
    residuals = [torch.arange(6.).reshape(3, 2), torch.tensor([[7.]])]
    jacobians = [torch.arange(12.).reshape(6, 2), torch.ones(1, 2)]
    flat, block, jac = RobustModel(nn.Identity()).normalize_RWJ(residuals, None, jacobians)
    assert block is None
    torch.testing.assert_close(flat, torch.cat([r.flatten() for r in residuals]))
    torch.testing.assert_close(jac, torch.cat(jacobians))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_weight_broadcast_gradients_match_direct_matrix_products(dtype):
    residual = torch.arange(24, dtype=dtype).reshape(2, 3, 2, 2).requires_grad_()
    weight = matrix_weights((2, 1, 2), 2, dtype).requires_grad_()
    flat, block, _ = RobustModel(nn.Identity()).normalize_RWJ(
        [residual], [weight], [torch.ones(24, 1, dtype=dtype)]
    )
    actual = torch.autograd.grad((block @ flat).square().sum(), (residual, weight))
    reference = torch.autograd.grad(
        (weight @ residual.unsqueeze(-1)).square().sum(), (residual, weight)
    )
    for computed, expected in zip(actual, reference):
        torch.testing.assert_close(computed, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("weight_at_step", [False, True])
@pytest.mark.parametrize("algorithm", ["LM", "GN"])
def test_fit_matches_batched_weighted_normal_equations(dtype, weight_at_step, algorithm):
    class SharedEstimate(nn.Module):
        def __init__(self):
            super().__init__()
            self.estimate = nn.Parameter(torch.zeros(2, dtype=dtype))

        def forward(self, observations):
            return self.estimate - observations

    observations = torch.arange(24, dtype=dtype).reshape(2, 3, 2, 2) / 10 + 2
    weight = matrix_weights((2, 1, 2), 2, dtype)
    damping = 1e-6
    normal = torch.zeros(2, 2, dtype=dtype)
    rhs = torch.zeros(2, dtype=dtype)
    # Each observation contributes its own matrix, with the middle axis broadcast.
    for outer in range(2):
        for row in range(3):
            for inner in range(2):
                matrix = weight[outer, 0, inner]
                if algorithm == "GN":
                    # GN applies the supplied matrix to both J and R before solving.
                    matrix = matrix.mT @ matrix
                normal += matrix
                rhs += matrix @ observations[outer, row, inner]
    if algorithm == "LM":
        normal.diagonal().mul_(1 + damping)
    expected = torch.linalg.solve(normal, rhs)
    model = SharedEstimate()
    kwargs = {"weight": None if weight_at_step else weight}
    if algorithm == "LM":
        kwargs["strategy"] = pp.optim.strategy.Constant(damping=damping)
    optimizer = getattr(pp.optim, algorithm)(model, **kwargs)

    before = model(observations).square().sum().item()
    loss = optimizer.step(observations, weight=weight if weight_at_step else None)

    assert loss < before
    torch.testing.assert_close(model.estimate, expected)
