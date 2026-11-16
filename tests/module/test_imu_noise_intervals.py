import pytest
import torch
import pypose as pp


def no_motion_covariance(dt, gyro_cov, acc_cov, initial):
    transition = torch.eye(9, dtype=dt.dtype).expand(initial.shape).clone()
    transition[..., 6:9, 3:6] = torch.eye(3, dtype=dt.dtype) * dt
    noise = torch.zeros_like(initial)
    noise[..., 0:3, 0:3] = torch.diag_embed(gyro_cov * dt)
    noise[..., 3:6, 3:6] = torch.diag_embed(acc_cov * dt)
    cross = torch.diag_embed(acc_cov * dt * dt * 0.5)
    noise[..., 3:6, 6:9] = noise[..., 6:9, 3:6] = cross
    noise[..., 6:9, 6:9] = torch.diag_embed(acc_cov * dt * dt * dt * 0.25)
    return transition @ initial @ transition.mT + noise


@pytest.mark.parametrize('dtype,interval', (
    (torch.float32, 0.), (torch.float64, 0.),
    (torch.float32, 1e-30), (torch.float64, 1e-200),
    (torch.float32, 0.01), (torch.float64, 0.01),
))
@pytest.mark.parametrize('batch', (1, 2))
@pytest.mark.parametrize('initial_variance', (0., 0.2))
def test_no_motion_covariance_matches_analytic_interval_scaling(dtype, interval, batch, initial_variance):
    gyro_cov = torch.tensor([1., 2., 3.], dtype=dtype)
    acc_cov = torch.tensor([4., 5., 6.], dtype=dtype)
    dt = torch.tensor(interval, dtype=dtype)
    initial = torch.eye(9, dtype=dtype).expand(batch, 9, 9) * initial_variance
    original = initial.clone()
    state = {'pos': torch.zeros(batch, 3, dtype=dtype), 'rot': pp.identity_SO3(batch, dtype=dtype),
             'vel': torch.zeros(batch, 3, dtype=dtype), 'cov': initial}
    imu = pp.module.IMUPreintegrator(gravity=0., reset=True).to(dtype=dtype)
    output = imu(dt.expand(batch, 1, 1), torch.zeros(batch, 1, 3, dtype=dtype),
                 torch.zeros(batch, 1, 3, dtype=dtype), init_state=state,
                 gyro_cov=gyro_cov, acc_cov=acc_cov)
    expected = no_motion_covariance(dt, gyro_cov, acc_cov, initial)
    assert torch.isfinite(output['cov']).all()
    torch.testing.assert_close(output['cov'], expected, atol=0, rtol=1e-6 if dtype == torch.float32 else 1e-14)
    torch.testing.assert_close(initial, original)


@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
def test_zero_duration_does_not_poison_stateful_future_updates(dtype):
    imu = pp.module.IMUPreintegrator(gravity=0., gyro_cov=0.1, acc_cov=0.2).to(dtype=dtype)
    control = pp.module.IMUPreintegrator(gravity=0., gyro_cov=0.1, acc_cov=0.2).to(dtype=dtype)
    gyro = torch.tensor([0.1, -0.2, 0.3], dtype=dtype)
    acceleration = torch.tensor([0.5, -0.1, 0.2], dtype=dtype)
    first = imu(torch.tensor([0.01], dtype=dtype), gyro, acceleration)
    control(torch.tensor([0.01], dtype=dtype), gyro, acceleration)
    zero = imu(torch.zeros(1, dtype=dtype), gyro, acceleration)
    for key in ('pos', 'vel', 'rot', 'cov', 'Rij'):
        a = zero[key].tensor() if isinstance(zero[key], pp.LieTensor) else zero[key]
        b = first[key].tensor() if isinstance(first[key], pp.LieTensor) else first[key]
        torch.testing.assert_close(a, b)
    actual = imu(torch.tensor([0.02], dtype=dtype), gyro, acceleration)
    expected = control(torch.tensor([0.02], dtype=dtype), gyro, acceleration)
    for key in ('pos', 'vel', 'rot', 'cov', 'Rij'):
        a = actual[key].tensor() if isinstance(actual[key], pp.LieTensor) else actual[key]
        b = expected[key].tensor() if isinstance(expected[key], pp.LieTensor) else expected[key]
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize('interval', (0., 0.01, 0.2))
def test_noise_parameter_gradients_match_analytic_covariance(interval):
    dt = torch.tensor(interval, dtype=torch.float64)
    gyro_cov = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64, requires_grad=True)
    acc_cov = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float64, requires_grad=True)
    imu = pp.module.IMUPreintegrator(gravity=0., reset=True).double()
    output = imu(dt.reshape(1), torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64),
                 gyro_cov=gyro_cov, acc_cov=acc_cov)['cov']
    expected = no_motion_covariance(dt, gyro_cov, acc_cov, torch.zeros(1, 9, 9, dtype=torch.float64))
    weights = torch.arange(81, dtype=torch.float64).reshape(9, 9) / 81
    actual_gradients = torch.autograd.grad((output * weights).sum(), (gyro_cov, acc_cov))
    expected_gradients = torch.autograd.grad((expected * weights).sum(), (gyro_cov, acc_cov))
    for actual, reference in zip(actual_gradients, expected_gradients):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, reference, atol=1e-16, rtol=1e-14)
