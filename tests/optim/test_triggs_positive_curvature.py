import pytest
import torch
import pypose as pp
from torch import nn
from pypose.optim.corrector import Triggs
from pypose.optim.kernel import Cauchy


class SquaredCost(nn.Module):
    def forward(self, x):
        return x.square()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch", [(), (3,), (2, 3)])
def test_triggs_preserves_positive_curvature_normal_equations(dtype, batch):
    shape = batch + (3,)
    R = torch.arange(torch.Size(shape).numel(), dtype=dtype).reshape(shape) / 5 - 1
    J = torch.arange(R.numel() * 2, dtype=dtype).reshape(-1, 2) / 7 - 1
    kernel = SquaredCost()
    corrected_R, corrected_J = Triggs(kernel)(R, J)

    # An affine residual makes the exact cost Hessian equal the robust
    # Gauss-Newton matrix, providing an independent autograd reference.
    def cost(step):
        residual = R + (J @ step).reshape_as(R)
        return kernel(residual.square().sum(-1)).sum() / 2

    zero = torch.zeros(2, dtype=dtype)
    gradient = torch.autograd.functional.jacobian(cost, zero)
    hessian = torch.autograd.functional.hessian(cost, zero)
    torch.testing.assert_close(corrected_J.T @ corrected_R.reshape(-1), gradient)
    torch.testing.assert_close(corrected_J.T @ corrected_J, hessian)
    expected_R = (2 * R.square().sum(-1, keepdim=True) / 3).sqrt() * R
    torch.testing.assert_close(corrected_R, expected_R)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_triggs_keeps_zero_residuals_and_negative_curvature_fallback(dtype):
    R = torch.tensor([[0., 0.], [0.5, -2.]], dtype=dtype)
    J = torch.arange(8, dtype=dtype).reshape(4, 2)
    corrected_R, corrected_J = Triggs(SquaredCost())(R, J)
    torch.testing.assert_close(corrected_R[0], torch.zeros(2, dtype=dtype))
    torch.testing.assert_close(corrected_J[:2], torch.zeros(2, 2, dtype=dtype))
    assert torch.isfinite(corrected_R).all()
    assert torch.isfinite(corrected_J).all()

    corrected_R, corrected_J = Triggs(Cauchy())(R, J)
    scale = (1 + R.square().sum(-1, keepdim=True)).rsqrt()
    torch.testing.assert_close(corrected_R, scale * R)
    torch.testing.assert_close(corrected_J, (scale.unsqueeze(-1) * J.view(2, 2, 2)).view_as(J))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("initial", [0., 3.])
def test_gauss_newton_triggs_follows_positive_curvature_step(dtype, initial):
    class ResidualModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.value = nn.Parameter(torch.tensor([initial], dtype=dtype))

        def forward(self, target):
            error = self.value - target
            return torch.stack((error, 2 * error), dim=-1)

    model = ResidualModel()
    target = torch.tensor(2., dtype=dtype)
    kernel = SquaredCost()
    optimizer = pp.optim.GN(model, kernel=kernel, corrector=Triggs(kernel))
    previous_loss = kernel(model(target).square().sum(-1)).sum()

    # The cost is 25 * (value - target)**4, so each exact GN update
    # removes one third of the error for either residual sign.
    for iteration in range(1, 5):
        loss = optimizer.step(target)
        expected = target + (initial - target) * (2 / 3) ** iteration
        torch.testing.assert_close(model.value, expected.reshape(1))
        torch.testing.assert_close(loss, 25 * (expected - target).pow(4))
        assert loss < previous_loss
        previous_loss = loss
