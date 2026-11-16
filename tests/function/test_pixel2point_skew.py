import pytest
import torch
import pypose as pp


@pytest.mark.parametrize('skew', [-20., 0., 20.])
def test_pixel2point_skew(skew):
    pixels = torch.tensor([[115., 100.], [50., -20.]], dtype=torch.float64)
    depth = torch.tensor([4., 2.], dtype=pixels.dtype)
    intrinsics = torch.tensor([
        [[100., skew, 50.], [0., 80., 40.], [0., 0., 1.]],
        [[90., -2 * skew, 30.], [0., 70., 20.], [0., 0., 1.]],
    ], dtype=pixels.dtype)
    expected = torch.linalg.solve(
        intrinsics.unsqueeze(-3), pp.cart2homo(pixels).unsqueeze(-1)
    ).squeeze(-1) * depth[..., None]

    actual = pp.pixel2point(pixels, depth, intrinsics)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(pp.point2pixel(actual, intrinsics), pixels.expand(2, -1, -1))


def test_pixel2point_skew_gradient():
    # Even at zero skew, its derivative must participate in calibration.
    skew = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    intrinsics = torch.eye(3, dtype=skew.dtype)
    intrinsics[0, 0], intrinsics[1, 1] = 100., 80.
    intrinsics[0, 1] = skew
    pixels = torch.tensor([[115., 100.]], dtype=skew.dtype, requires_grad=True)
    depth = torch.tensor([4.], dtype=skew.dtype, requires_grad=True)
    expected = torch.linalg.solve(intrinsics, pp.cart2homo(pixels).mT).mT * depth[:, None]
    actual = pp.pixel2point(pixels, depth, intrinsics)

    inputs = (skew, pixels, depth)
    expected_grad = torch.autograd.grad(expected.sum(), inputs, retain_graph=True)
    actual_grad = torch.autograd.grad(actual.sum(), inputs)
    torch.testing.assert_close(actual, expected)
    for actual_value, expected_value in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual_value, expected_value)
