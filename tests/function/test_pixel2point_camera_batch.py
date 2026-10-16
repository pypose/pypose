import pytest
import torch
import pypose as pp


@pytest.mark.parametrize('shared_depth,count', [(False, 2), (False, 3), (True, 0), (True, 3)])
def test_pixel2point_camera_broadcast(shared_depth, count):
    # Equal camera/point counts silently mix axes; unequal counts used to fail.
    pixels = torch.tensor([[3, 4], [5, 6], [2, -1]], dtype=torch.float64)[:count]
    depth = torch.tensor([2, 3, 4], dtype=torch.float64)[:count]
    intrinsics = torch.tensor([
        [[2, 0, 1], [0, 3, 2], [0, 0, 1]],
        [[4, 0, -1], [0, 5, 1], [0, 0, 1]],
    ], dtype=torch.float64)
    if not shared_depth:
        pixels = torch.stack([pixels, pixels + 1])
        depth = torch.stack([depth, depth + 1])

    # Solve camera rays independently of the elementwise implementation.
    homogeneous = pp.cart2homo(pixels)
    expected = torch.linalg.solve(
        intrinsics.unsqueeze(-3), homogeneous.unsqueeze(-1)
    ).squeeze(-1) * depth[..., None]
    actual = pp.pixel2point(pixels, depth, intrinsics)

    assert actual.shape == (2, count, 3)
    torch.testing.assert_close(actual, expected)
