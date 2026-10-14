import math

import pytest
import torch
import pypose as pp



@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('count', [2, 3, 5])
def test_distinct_camera_intrinsics_and_round_trip(dtype, count):
    intrinsics = torch.tensor([[[2, 0, 3], [0, 4, 1], [0, 0, 1]], [[5, 0, -2], [0, 3, 4], [0, 0, 1]], [[3, 0, 1], [0, 6, -3], [0, 0, 1]]], dtype=dtype)
    rays = torch.tensor([[[-1, 0.5, 1], [0.2, -0.3, 1], [0.7, 0.2, 1], [0.8, 0.3, 1], [-0.2, 0.9, 1]]], dtype=dtype)[:, :count].expand(3, -1, -1)
    depth = torch.arange(1, 3 * count + 1, dtype=dtype).reshape(3, count)
    expected = rays * depth[..., None]
    camera_homogeneous = rays @ intrinsics.mT
    pixels = camera_homogeneous[..., :2]
    actual = pp.pixel2point(pixels, depth, intrinsics)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(pp.point2pixel(actual, intrinsics), pixels)


def test_broadcast_cameras_with_leading_batch_dimensions():
    pixels = torch.tensor([[[[3.0, 4.0], [5.0, 6.0], [2.0, -1.0]]]], dtype=torch.float64).expand(2, 1, 3, 2)
    intrinsics = torch.tensor([[[[2.0, 0, 1], [0, 3.0, 2], [0, 0, 1]], [[4.0, 0, -1], [0, 5.0, 1], [0, 0, 1]]]], dtype=torch.float64)
    depth = torch.ones((2, 2, 3), dtype=torch.float64) * 2
    actual = pp.pixel2point(pixels, depth, intrinsics)
    expected = torch.linalg.solve(intrinsics.unsqueeze(-3), pp.cart2homo(pixels).unsqueeze(-1)).squeeze(-1) * depth[..., None]
    assert actual.shape == (2, 2, 3, 3)
    torch.testing.assert_close(actual, expected)


def test_unbatched_intrinsics_empty_points_and_gradient():
    k = torch.tensor([[2.0, 0, 0.3], [0, 3.0, -0.4], [0, 0, 1]], dtype=torch.float64)
    torch.testing.assert_close(pp.pixel2point(torch.empty(2, 0, 2, dtype=torch.float64), torch.empty(2, 0, dtype=torch.float64), k), torch.empty(2, 0, 3, dtype=torch.float64))
    pixels = torch.tensor([[[0.1, 0.2], [0.3, 0.4], [0.5, -0.7]], [[-0.2, 0.1], [0.8, 0.5], [-0.4, 0.6]]], dtype=torch.float64, requires_grad=True)
    depth = torch.tensor([[1.2, 2.3, 3.1], [2.1, 3.2, 4.3]], dtype=torch.float64, requires_grad=True)
    k = k.expand(2, 3, 3).clone().requires_grad_()
    assert torch.autograd.gradcheck(pp.pixel2point, (pixels, depth, k))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('count', [0, 3])
@pytest.mark.parametrize('leading_batches', [False, True])
def test_shared_depth_broadcasts_over_camera_batches(dtype, count, leading_batches):
    pixels = torch.tensor([[3, 4], [5, 6], [2, -1]], dtype=dtype)[:count]
    depth = torch.tensor([2, 3, 4], dtype=dtype)[:count]
    intrinsics = torch.tensor([
        [[2, 0, 1], [0, 3, 2], [0, 0, 1]],
        [[4, 0, -1], [0, 5, 1], [0, 0, 1]],
    ], dtype=dtype)
    if leading_batches:
        pixels = pixels.reshape(1, 1, count, 2).expand(3, 1, count, 2)
        intrinsics = intrinsics.unsqueeze(0)
        expected_shape = (3, 2, count, 3)
    else:
        expected_shape = (2, count, 3)

    homogeneous = torch.cat([pixels, torch.ones_like(pixels[..., :1])], dim=-1)
    expected = torch.linalg.solve(
        intrinsics.unsqueeze(-3), homogeneous.unsqueeze(-1)
    ).squeeze(-1) * depth[..., None]
    actual = pp.pixel2point(pixels, depth, intrinsics)

    assert actual.shape == expected_shape
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual[..., 2], depth.expand(expected_shape[:-1]))
    torch.testing.assert_close(
        pp.point2pixel(actual, intrinsics), pixels.expand(*expected_shape[:-1], 2)
    )


def test_shared_depth_camera_broadcast_gradient():
    pixels = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, -0.7]],
                          dtype=torch.float64, requires_grad=True)
    depth = torch.tensor([1.2, 2.3, 3.1], dtype=torch.float64, requires_grad=True)
    intrinsics = torch.tensor([
        [[2, 0, 0.3], [0, 3, -0.4], [0, 0, 1]],
        [[4, 0, -0.2], [0, 5, 0.6], [0, 0, 1]],
    ], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(pp.pixel2point, (pixels, depth, intrinsics))
