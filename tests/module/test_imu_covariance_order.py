import pytest
import torch
import pypose as pp


@pytest.mark.parametrize('batch', [1, 2])
@pytest.mark.parametrize('frames', [3, 7])
@pytest.mark.parametrize('initial_covariance', [False, True])
def test_covariance_matches_sequential_updates(batch, frames, initial_covariance):
    torch.manual_seed(42)
    dtype = torch.float64
    dt = torch.full((batch, frames, 1), 0.1, dtype=dtype)
    rotations = pp.so3(torch.randn(batch, frames, 3, dtype=dtype) * 0.3).Exp()
    accumulated = []
    rotation = pp.identity_SO3(batch, dtype=dtype)
    for k in range(frames):
        rotation = rotation * rotations[:, k]
        accumulated.append(rotation)
    cov_input = {
        'dt': dt,
        'Rk': rotations,
        'Rij': torch.stack(accumulated, dim=1),
        'Ha': pp.vec2skew(torch.randn(batch, frames, 3, dtype=dtype)),
    }
    covariance = torch.zeros(batch, 9, 9, dtype=dtype)
    if initial_covariance:
        factor = torch.randn(batch, 9, 9, dtype=dtype)
        covariance = factor @ factor.mT * 0.01
    gyro_cov = torch.full((batch, 1, 3), 0.01, dtype=dtype)
    acc_cov = torch.full((batch, 1, 3), 0.02, dtype=dtype)

    expected = covariance
    for k in range(frames):
        single = {key: value[:, k:k+1] for key, value in cov_input.items()}
        expected = pp.module.IMUPreintegrator.propagate_cov(
            single, expected, gyro_cov, acc_cov)['cov']
    actual = pp.module.IMUPreintegrator.propagate_cov(
        cov_input, covariance, gyro_cov, acc_cov)['cov']
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
