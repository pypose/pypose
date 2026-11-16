import pytest
import torch
import pypose as pp


def camera_problem(count, dtype, seed=0, batch=()):
    generator = torch.Generator().manual_seed(seed)
    points = torch.randn((*batch, count, 3), dtype=dtype, generator=generator)
    pose = pp.SE3(torch.tensor([.2, -.1, 8., 0., 0., 0., 1.], dtype=dtype))
    intrinsics = torch.tensor(
        [[400., 0., 320.], [0., 450., 240.], [0., 0., 1.]], dtype=dtype
    ).expand(*batch, 3, 3)
    pixels = pp.point2pixel(points, intrinsics, pose)
    return points, pixels, intrinsics, pose


def projective_matrix(pixels, alpha, intrinsics):
    rows = []
    for point in range(pixels.shape[-2]):
        u, v = pixels[..., point, 0], pixels[..., point, 1]
        horizontal = torch.stack(
            (intrinsics[..., 0, 0], torch.zeros_like(u), intrinsics[..., 0, 2] - u),
            dim=-1,
        )
        vertical = torch.stack(
            (torch.zeros_like(v), intrinsics[..., 1, 1], intrinsics[..., 1, 2] - v),
            dim=-1,
        )
        for constraint in (horizontal, vertical):
            rows.append(
                (alpha[..., point, :, None] * constraint[..., None, :]).flatten(-2)
            )
    return torch.stack(rows, dim=-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("count", [4, 5, 9])
@pytest.mark.parametrize("batch", [(), (2, 1)])
def test_selected_vectors_span_smallest_real_subspace(dtype, count, batch):
    points, pixels, K, _ = camera_problem(count, dtype, batch=batch)
    solver = pp.module.EPnP(refine=False)
    alpha = solver._compute_alpha(points, solver._svd_basis(points))
    vectors = solver._compute_nullv(pixels, alpha, K)
    assert torch.all(torch.linalg.matrix_rank(vectors) == 4)
    matrix = projective_matrix(pixels, alpha, K).double()
    _, _, vh = torch.linalg.svd(matrix, full_matrices=True)
    expected = vh[..., -4:, :]
    basis = torch.linalg.qr(vectors.double().mT).Q
    torch.testing.assert_close(
        basis @ basis.mT, expected.mT @ expected, rtol=0, atol=2e-3
    )
    assert vectors.dtype == dtype
    assert vectors.shape == (*batch, 4, 12)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("refine", [False, True])
@pytest.mark.parametrize("seed", [0, 2, 5])
def test_five_point_reprojection(dtype, refine, seed):
    points, pixels, K, _ = camera_problem(5, dtype, seed)
    result = pp.module.EPnP(refine=refine)(points, pixels, K)
    error = pp.reprojerr(points, pixels, K, result, reduction="norm")
    assert torch.isfinite(error).all()
    assert error.mean() < (0.05 if dtype == torch.float32 else 1e-6)


@pytest.mark.parametrize("seed", [1, 3])
def test_preserves_successful_four_point_refinement(seed):
    points, pixels, K, _ = camera_problem(4, torch.float64, seed)
    result = pp.module.EPnP(refine=True)(points, pixels, K)
    error = pp.reprojerr(points, pixels, K, result, reduction="norm")
    assert error.mean() < 1e-7


def test_mixed_batch_matches_independent_subspaces():
    problems = [camera_problem(5, torch.float32, seed) for seed in (0, 1, 2)]
    points = torch.stack([p[0] for p in problems])
    pixels = torch.stack([p[1] for p in problems])
    K = torch.stack([p[2] for p in problems])
    solver = pp.module.EPnP(refine=False)
    alpha = solver._compute_alpha(points, solver._svd_basis(points))
    together = solver._compute_nullv(pixels, alpha, K)
    for index in range(3):
        separate = solver._compute_nullv(pixels[index], alpha[index], K[index])
        assert torch.linalg.matrix_rank(together[index]) == 4
        left = torch.linalg.qr(together[index].double().mT).Q
        right = torch.linalg.qr(separate.double().mT).Q
        torch.testing.assert_close(left @ left.mT, right @ right.mT, rtol=0, atol=2e-3)


@pytest.mark.parametrize("least", [0, 1, 4, 12])
def test_requested_subspace_width(least):
    points, pixels, K, _ = camera_problem(8, torch.float64)
    solver = pp.module.EPnP(refine=False)
    alpha = solver._compute_alpha(points, solver._svd_basis(points))
    result = solver._compute_nullv(pixels, alpha, K, least=least)
    assert result.shape == (least, 12)
    assert torch.isfinite(result).all()


def test_subspace_backward_is_finite():
    points, pixels, K, _ = camera_problem(5, torch.float32)
    pixels.requires_grad_()
    solver = pp.module.EPnP(refine=False)
    alpha = solver._compute_alpha(points, solver._svd_basis(points))
    vectors = solver._compute_nullv(pixels, alpha, K)
    basis = torch.linalg.qr(vectors.mT).Q
    projector = basis @ basis.mT
    loss = (projector * torch.arange(144).reshape(12, 12)).sum()
    gradient = torch.autograd.grad(loss, pixels)[0]
    assert torch.isfinite(gradient).all()
    assert gradient.abs().max() > 0


def test_mixed_batch_subspace_backward_is_finite():
    problems = [camera_problem(5, torch.float32, seed) for seed in (0, 1, 2)]
    points = torch.stack([p[0] for p in problems])
    pixels = torch.stack([p[1] for p in problems]).requires_grad_()
    K = torch.stack([p[2] for p in problems])
    solver = pp.module.EPnP(refine=False)
    alpha = solver._compute_alpha(points, solver._svd_basis(points))
    vectors = solver._compute_nullv(pixels, alpha, K)
    basis = torch.linalg.qr(vectors.mT).Q
    projector = basis @ basis.mT
    loss = (projector * torch.arange(144).reshape(12, 12)).sum()
    gradient = torch.autograd.grad(loss, pixels)[0]
    assert torch.isfinite(gradient).all()
    assert (gradient.abs().amax(dim=(-1, -2)) > 0).all()

