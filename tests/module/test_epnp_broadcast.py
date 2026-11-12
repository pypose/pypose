import pytest
import torch
import pypose as pp


@pytest.mark.parametrize("refine", [False, True])
@pytest.mark.parametrize("shared", ["points", "pixels", "calibration_grid"])
def test_epnp_broadcast_batches_match_independent_pose_recovery(refine, shared):
    camera_points = torch.tensor([
        [0., 0., 4.], [1., 0., 5.], [0., 1., 6.], [1., 1., 4.],
        [-1., .5, 5.], [.5, -1., 6.], [1.5, -.5, 4.5], [-.5, 1.5, 5.5],
    ], dtype=torch.float64)
    intrinsics = torch.tensor([[400., 0., 320.], [0., 450., 240.], [0., 0., 1.]],
                              dtype=torch.float64)
    poses = pp.SE3(torch.tensor([
        [.2, .1, .3, 0., 0., 0., 1.], [-.2, .2, .5, 0., 0., 0., 1.],
    ], dtype=torch.float64))
    if shared == "points":
        points = camera_points
        pixels = pp.point2pixel(points, intrinsics, poses)
        expected = poses
    elif shared == "pixels":
        points = poses.Inv().unsqueeze(-2) @ camera_points
        pixels = pp.point2pixel(camera_points, intrinsics)
        expected = poses
    else:
        points = (poses.Inv().unsqueeze(-2) @ camera_points).unsqueeze(1)
        intrinsics = intrinsics.repeat(3, 1, 1)
        intrinsics[:, 0, 0] = torch.tensor([300., 400., 500.])
        intrinsics[:, 1, 1] = torch.tensor([350., 450., 550.])
        pixels = pp.point2pixel(camera_points, intrinsics)
        expected = poses.unsqueeze(1).expand(2, 3, 7)

    solver = pp.module.EPnP(refine=refine)
    actual = solver(points, pixels, intrinsics)
    assert actual.lshape == expected.lshape
    torch.testing.assert_close(actual.matrix(), expected.matrix(), rtol=1e-5, atol=1e-5)

    batch = expected.lshape
    expanded_points = points.expand(*batch, *points.shape[-2:]).reshape(-1, *points.shape[-2:])
    expanded_pixels = pixels.expand(*batch, *pixels.shape[-2:]).reshape(-1, *pixels.shape[-2:])
    expanded_intrinsics = intrinsics.expand(*batch, 3, 3).reshape(-1, 3, 3)
    independent = torch.stack([
        solver(point, pixel, calibration).matrix()
        for point, pixel, calibration in zip(expanded_points, expanded_pixels, expanded_intrinsics)
    ]).reshape(*batch, 4, 4)
    torch.testing.assert_close(actual.matrix(), independent, rtol=1e-5, atol=1e-5)
