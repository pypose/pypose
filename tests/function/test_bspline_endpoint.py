import pytest
import torch
import pypose as pp


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('batch', [(), (2,), (2, 3)])
@pytest.mark.parametrize('extrapolate', [False, True])
def test_bspline_endpoint_cubic_basis(dtype, batch, extrapolate):
    # Translation and rotation along the same axis commute. The ordinary
    # cubic B-spline basis therefore gives an independent endpoint oracle.
    coordinates = torch.zeros(batch + (9, 6), dtype=dtype)
    t = torch.linspace(-0.7, 0.8, 9, dtype=dtype)
    coordinates[..., 2] = t.square()
    coordinates[..., 5] = t.pow(3)
    data = pp.se3(coordinates).Exp()
    if extrapolate:
        expected = data[..., -1, :]
    else:
        endpoint = (coordinates[..., -3, :] + 4 * coordinates[..., -2, :]
                    + coordinates[..., -1, :]) / 6
        expected = pp.se3(endpoint).Exp()
    actual = pp.bspline(data, interval=0.3, extrapolate=extrapolate)
    pp.testing.assert_close(actual[..., -1, :], expected)


@pytest.mark.parametrize('count', [1, 2, 4])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_bspline_endpoint_extrapolated_short_trajectory(count, dtype):
    torch.manual_seed(11)
    data = pp.randn_SE3(2, count, dtype=dtype)
    actual = pp.bspline(data, interval=0.4, extrapolate=True)
    pp.testing.assert_close(actual[..., -1, :], data[..., -1, :])


@pytest.mark.parametrize('batch', [(), (2, 3)])
@pytest.mark.parametrize('extrapolate', [False, True])
def test_bspline_endpoint_depends_only_on_final_controls(batch, extrapolate):
    torch.manual_seed(19)
    coordinates = (0.15 * torch.randn(batch + (18, 6), dtype=torch.float64))
    coordinates.requires_grad_()
    # Keep a genuinely strided pose sequence through the public API.
    data = pp.se3(coordinates).Exp()[..., ::2, :]
    assert not data.is_contiguous()
    full = pp.bspline(data, interval=0.4, extrapolate=extrapolate)
    tail = pp.bspline(data[..., -4:, :], interval=0.4,
                      extrapolate=extrapolate)
    pp.testing.assert_close(full[..., -1, :], tail[..., -1, :])
    weights = torch.arange(1, 8, dtype=coordinates.dtype)
    full_loss = (full[..., -1, :].tensor() * weights).sum()
    tail_loss = (tail[..., -1, :].tensor() * weights).sum()
    full_grad, = torch.autograd.grad(full_loss, coordinates, retain_graph=True)
    tail_grad, = torch.autograd.grad(tail_loss, coordinates)
    torch.testing.assert_close(full_grad, tail_grad)
    assert torch.count_nonzero(full_grad[..., :-8, :]) == 0
