import pytest
import torch
import pypose as pp


def camera_problem(batch, count, dtype):
    generator = torch.Generator().manual_seed(613)
    points = torch.randn(batch + (count, 3), generator=generator, dtype=dtype)
    points = points * torch.tensor([1., .7, .5], dtype=dtype)
    rotation = pp.so3(torch.tensor([.05, -.08, .02], dtype=dtype)).Exp().matrix()
    translation = torch.tensor([.2, -.1, 5.], dtype=dtype)
    intrinsics = torch.tensor([[450., 0., 320.], [0., 440., 240.],
                               [0., 0., 1.]], dtype=dtype)
    camera = (rotation @ points.mT).mT + translation
    projected = (intrinsics @ camera.mT).mT
    pixels = projected[..., :2] / projected[..., 2:]
    return points, pixels, intrinsics, rotation, translation


def projective_matrix(pixels, alpha, intrinsics):
    """Build the two camera equations for each point using control-point blocks."""
    u, v = pixels.unbind(-1)
    zero = torch.zeros_like(u)
    fu = intrinsics[..., 0, 0, None].expand_as(u)
    fv = intrinsics[..., 1, 1, None].expand_as(v)
    horizontal = torch.stack((fu, zero, intrinsics[..., 0, 2, None] - u), -1)
    vertical = torch.stack((zero, fv, intrinsics[..., 1, 2, None] - v), -1)
    rows = torch.stack((horizontal, vertical), -2)
    blocks = rows.unsqueeze(-2) * alpha.unsqueeze(-2).unsqueeze(-1)
    return blocks.flatten(-2).flatten(-3, -2)


@pytest.mark.parametrize('batch', [(), (2,), (2, 3)])
@pytest.mark.parametrize('count', [16, 48])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_epnp_projective_subspace(batch, count, dtype):
    points, pixels, intrinsics, _, _ = camera_problem(batch, count, dtype)
    bases = pp.module.EPnP._svd_basis(points)
    alpha = pp.module.EPnP._compute_alpha(points, bases)
    actual = pp.module.EPnP._compute_nullv(pixels, alpha, intrinsics)
    matrix = projective_matrix(pixels, alpha, intrinsics)
    _, _, vh = torch.linalg.svd(matrix, full_matrices=False)
    expected = vh[..., -4:, :]
    tolerance = 2e-3 if dtype == torch.float32 else 2e-10
    torch.testing.assert_close(actual.mT @ actual, expected.mT @ expected,
                               atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize('batch', [(), (2,), (2, 3)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('refine', [False, True])
def test_epnp_recovers_camera_pose(batch, dtype, refine):
    points, pixels, intrinsics, rotation, translation = camera_problem(batch, 32, dtype)
    pose = pp.module.EPnP(intrinsics, refine=refine)(points, pixels)
    matrix = pose.matrix()
    tolerance = 2e-3 if dtype == torch.float32 else 2e-7
    torch.testing.assert_close(matrix[..., :3, :3], rotation.expand(batch + (3, 3)),
                               atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(matrix[..., :3, 3], translation.expand(batch + (3,)),
                               atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize('count', [16, 48])
def test_epnp_projective_subspace_gradients(count):
    generator = torch.Generator().manual_seed(173)
    pixels = torch.randn(2, count, 2, generator=generator, dtype=torch.float64)
    alpha = torch.randn(2, count, 4, generator=generator, dtype=torch.float64).softmax(-1)
    intrinsics = torch.tensor([[2.4, 0., 3.4], [0., 2.1, 1.8], [0., 0., 1.]],
                               dtype=torch.float64)
    inputs = tuple(value.requires_grad_() for value in (pixels, alpha, intrinsics))
    actual = pp.module.EPnP._compute_nullv(*inputs)
    _, _, vh = torch.linalg.svd(projective_matrix(*inputs), full_matrices=False)
    expected = vh[..., -4:, :]
    weight = torch.randn(2, 12, 12, generator=generator, dtype=torch.float64)
    loss = ((actual.mT @ actual) * weight).sum()
    reference_loss = ((expected.mT @ expected) * weight).sum()
    gradients = torch.autograd.grad(loss, inputs)
    expected_gradients = torch.autograd.grad(reference_loss, inputs)
    for result, reference in zip(gradients, expected_gradients):
        torch.testing.assert_close(result, reference, atol=2e-9, rtol=2e-8)
