import pytest
import torch
import pypose as pp


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
@pytest.mark.parametrize('frames', [2, 5])
@pytest.mark.parametrize('known_rot', [False, True])
@pytest.mark.parametrize('entry', ['constructor', 'forward', 'integrate_predict'])
def test_batched_initial_state(device, frames, known_rot, entry):
    dtype = torch.float64
    pos = torch.tensor([[1., 2., 3.], [4., 5., 6.]], dtype=dtype, device=device)
    vel = torch.tensor([[0.3, -0.2, 0.1], [-0.4, 0.5, -0.6]], dtype=dtype, device=device)
    rot = pp.so3(torch.tensor([[0., 0., 0.2], [0., 0., -0.4]],
                             dtype=dtype, device=device)).Exp()
    state = {'pos': pos, 'vel': vel, 'rot': rot}
    dt = torch.full((2, frames, 1), 0.01, dtype=dtype, device=device)
    gyro = torch.zeros(2, frames, 3, dtype=dtype, device=device)
    acc = torch.zeros_like(gyro)
    acc[..., 2] = 9.81007
    measurements = rot[:, None].expand(2, frames, 4) if known_rot else None
    module = pp.module.IMUPreintegrator(**state).to(device=device, dtype=dtype) \
        if entry == 'constructor' else pp.module.IMUPreintegrator().to(device=device, dtype=dtype)
    if entry == 'integrate_predict':
        increments = module.integrate(dt, gyro, acc, rot=measurements, init_rot=rot)
        output = module.predict(state, increments)
    else:
        output = module(dt, gyro, acc, rot=measurements,
                        init_state=state if entry == 'forward' else None)
    time = dt.cumsum(dim=1)
    torch.testing.assert_close(output['pos'], pos[:, None] + vel[:, None] * time)
    torch.testing.assert_close(output['vel'], vel[:, None].expand(2, frames, 3))
    torch.testing.assert_close(output['rot'].tensor(), rot[:, None].expand(2, frames, 4).tensor())
    assert state['pos'] is pos and state['vel'] is vel and state['rot'] is rot
    assert pos.shape == (2, 3) and rot.shape == (2, 4)


@pytest.mark.parametrize('state_shape', ['vector', 'singleton_frame'])
def test_existing_initial_state_and_unbatched_measurements(state_shape):
    pos = torch.tensor([1., 2., 3.])
    vel = torch.tensor([0.2, 0.4, 0.6])
    rot = pp.identity_SO3()
    state = {'pos': pos, 'vel': vel, 'rot': rot}
    if state_shape == 'singleton_frame':
        state = {key: value[None, None] for key, value in state.items()}
    dt = torch.full((4, 1), 0.01)
    gyro = torch.zeros(4, 3)
    acc = torch.tensor([0., 0., 9.81007]).expand(4, 3)
    output = pp.module.IMUPreintegrator()(dt, gyro, acc, init_state=state)
    torch.testing.assert_close(output['pos'], (pos + vel * dt.cumsum(0))[None])


@pytest.mark.parametrize('entry', ['constructor', 'forward'])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_initial_state_gradients(entry, device):
    pos = torch.ones(2, 3, device=device, requires_grad=True)
    vel = torch.zeros(2, 3, device=device, requires_grad=True)
    state = {'pos': pos, 'vel': vel, 'rot': pp.identity_SO3(2, device=device)}
    dt = torch.full((2, 4, 1), 0.01, device=device)
    gyro = torch.zeros(2, 4, 3, device=device)
    acc = torch.zeros_like(gyro)
    module = pp.module.IMUPreintegrator(**state).to(device) if entry == 'constructor' \
        else pp.module.IMUPreintegrator().to(device)
    output = module(dt, gyro, acc, init_state=state if entry == 'forward' else None)
    output['pos'].sum().backward()
    torch.testing.assert_close(pos.grad, torch.full_like(pos, 4.))
    torch.testing.assert_close(vel.grad, dt.cumsum(dim=1).sum(dim=1).expand_as(vel))
