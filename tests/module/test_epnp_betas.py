import pytest
import torch
import pypose as pp


def quadratic_estimates(batch, dtype):
    values = {
        3: [4., -2., 9.],
        6: [4., -2., 3., 9., -4., 16.],
        10: [4., -2., 9., 3., -4., 16., -5., 6., -7., 25.],
    }
    return {width: torch.tensor(value, dtype=dtype).expand(batch + (width,)).clone()
            for width, value in values.items()}


def reconstruct_betas(values, batch, dtype):
    model = pp.module.EPnP(torch.eye(3, dtype=dtype), refine=False)
    # Supply quadratic estimates to isolate their conversion into beta values.
    model.solver.forward = lambda matrix, rhs: values[matrix.shape[-1]]
    return model._compute_betas(torch.zeros(batch + (6, 10), dtype=dtype),
                                torch.ones(batch + (6,), dtype=dtype))


@pytest.mark.parametrize('batch', [(), (2, 3)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('zero', [0., -0.])
@pytest.mark.parametrize('width,index', [(3, 0), (3, 1), (6, 0), (6, 1), (6, 2),
                                         (10, 0), (10, 1), (10, 3), (10, 6)])
def test_epnp_zero_sign_preserves_beta_magnitudes(batch, dtype, zero, width, index):
    values = quadratic_estimates(batch, dtype)
    values[width][..., index] = zero
    betas = reconstruct_betas(values, batch, dtype)
    # Each diagonal quadratic estimate independently specifies a squared beta
    # magnitude, even when a coefficient used to choose its direction is zero.
    for candidate, positions, diagonals in [(1, (2, 3), (0, 2)),
                                             (2, (1, 2, 3), (0, 3, 5))]:
        estimates = values[(3, 6, 10)[candidate - 1]]
        torch.testing.assert_close(betas[candidate, ..., positions].square(),
                                   estimates[..., diagonals].abs(), rtol=0, atol=0)
    # Check the four-vector magnitudes independently of coefficient ordering.
    actual = betas[3].square().sort(dim=-1).values
    expected = values[10][..., (0, 2, 5, 9)].abs().sort(dim=-1).values
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('batch', [(), (2, 3)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('anchor_sign', [1, -1])
def test_epnp_nonzero_beta_directions(batch, dtype, anchor_sign):
    values = quadratic_estimates(batch, dtype)
    for value in values.values():
        value[..., 0] *= anchor_sign
    actual = reconstruct_betas(values, batch, dtype)
    expected = torch.tensor([[0., 0., 0., 1.], [0., 0., 2., -3. * anchor_sign],
                             [0., 2., -3. * anchor_sign, 4. * anchor_sign]], dtype=dtype)
    expected = expected.reshape((3,) + (1,) * len(batch) + (4,)).expand_as(actual[:3])
    torch.testing.assert_close(actual[:3], expected, rtol=0, atol=0)


@pytest.mark.parametrize('batch', [(), (2, 3)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_epnp_zero_estimates_do_not_create_magnitudes(batch, dtype):
    values = {width: torch.zeros_like(value)
              for width, value in quadratic_estimates(batch, dtype).items()}
    betas = reconstruct_betas(values, batch, dtype)
    assert torch.count_nonzero(betas[1:]) == 0
    assert torch.all(betas[0, ..., -1] == 1)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('refine', [False, True])
def test_epnp_camera_pose_with_replayed_zero_beta_anchor(dtype, refine):
    generator = torch.Generator().manual_seed(613)
    points = torch.randn(32, 3, generator=generator, dtype=dtype)
    points = points * torch.tensor([1., .7, .5], dtype=dtype)
    rotation = pp.so3(torch.tensor([.05, -.08, .02], dtype=dtype)).Exp().matrix()
    translation = torch.tensor([.2, -.1, 5.], dtype=dtype)
    intrinsics = torch.tensor([[450., 0., 320.], [0., 440., 240.],
                               [0., 0., 1.]], dtype=dtype)
    camera = (rotation @ points.mT).mT + translation
    projected = (intrinsics @ camera.mT).mT
    pixels = projected[..., :2] / projected[..., 2:]
    model = pp.module.EPnP(intrinsics, refine=refine)
    model.solver = pp.optim.solver.LSTSQ(driver='gelsd')
    solve = model.solver.forward

    def replay_zero_anchor(matrix, rhs):
        estimates = solve(matrix, rhs)
        if matrix.shape[-1] == 3:
            # Replay the signed-zero anchor observed with the default QR
            # solver; gelsd keeps the rest of this regression deterministic.
            estimates = estimates.clone()
            estimates[..., 0] = -0.
        return estimates

    model.solver.forward = replay_zero_anchor
    actual = model(points, pixels).matrix()
    tolerance = 2e-3 if dtype == torch.float32 else 2e-7
    torch.testing.assert_close(actual[:3, :3], rotation, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(actual[:3, 3], translation, atol=tolerance, rtol=tolerance)
