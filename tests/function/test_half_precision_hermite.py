import pytest
import torch
import pypose as pp


DEVICES = ('cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA is unavailable')))


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype,count', ((torch.float16, 3000), (torch.bfloat16, 300)))
@pytest.mark.parametrize('batch', ((), (2,)))
@pytest.mark.parametrize('interval', (0.1, 0.3))
def test_half_precision_trajectory_has_finite_values_and_gradients(device, dtype, count, batch, interval):
    t = torch.linspace(0, 4, count, device=device, dtype=torch.float64)
    source = torch.stack((t.sin(), (t/2).cos()), dim=-1).to(dtype)
    points = source.expand(batch + source.shape).clone().requires_grad_()
    original = points.detach().clone()
    result = pp.chspline(points, interval)
    reference_points = points.detach().double().requires_grad_()
    reference = pp.chspline(reference_points, interval)
    assert result.dtype == dtype
    assert result.device == points.device
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result[..., [0, -1], :], points[..., [0, -1], :])
    tolerance = 4 * torch.finfo(dtype).eps
    torch.testing.assert_close(result.double(), reference, atol=tolerance, rtol=tolerance)
    gradient, = torch.autograd.grad(result.float().sum(), points)
    expected_gradient, = torch.autograd.grad(reference.sum(), reference_points)
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient.double(), expected_gradient,
                               atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(points.detach(), original)


def test_large_float16_endpoint_difference_stays_finite():
    points = torch.tensor([[-60000.0], [60000.0]], dtype=torch.float16,
                           requires_grad=True)
    result = pp.chspline(points, 0.25)
    expected = torch.tensor([[-60000.0], [-30000.0], [0.0], [30000.0], [60000.0]],
                             dtype=torch.float16)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, expected)
    gradient, = torch.autograd.grad(result.float().sum(), points)
    torch.testing.assert_close(gradient, torch.full_like(points, 2.5))


@pytest.mark.parametrize('dtype', (torch.float16, torch.bfloat16, torch.float32, torch.float64))
def test_short_linear_spline_preserves_dtype_and_values(dtype):
    points = torch.tensor([[0.0, 1.0], [4.0, 5.0]], dtype=dtype)
    expected = torch.tensor([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0],
                              [3.0, 4.0], [4.0, 5.0]], dtype=dtype)
    torch.testing.assert_close(pp.chspline(points, 0.25), expected)


@pytest.mark.parametrize('dtype', (torch.float16, torch.bfloat16))
def test_empty_half_precision_batch_retains_shape_and_gradient(dtype):
    points = torch.empty(0, 4, 2, dtype=dtype, requires_grad=True)
    result = pp.chspline(points, 0.5)
    assert result.shape == (0, 7, 2)
    assert result.dtype == dtype
    result.float().sum().backward()
    assert points.grad is not None and points.grad.shape == points.shape
