from itertools import product

import pytest
import torch
import pypose as pp


DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


class NonlinearModel(pp.module.NLS):
    def __init__(self, weight, phase):
        super().__init__()
        self.weight = torch.nn.Parameter(weight.clone())
        self.register_buffer('phase', phase.clone())

    def state_transition(self, state, input, t=None):
        time = torch.as_tensor(0 if t is None else t, device=state.device, dtype=state.dtype)
        time = time.reshape(()) if time.numel() == 1 else time.unsqueeze(-1)
        return state + 0.1 * (self.weight * state + self.phase).sin() + input + 0.01 * time

    def observation(self, state, input, t=None):
        # Observation and state dimensions deliberately differ.
        return state[..., :1].square() + 0.3 * state[..., 1:2].sin() + 0.02 * input


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('batch', [(), (1,), (3,), (2, 3)])
@pytest.mark.parametrize('shared', [False, True])
def test_ekf_batch_matches_independent_updates(device, dtype, batch, shared):
    indices = list(product(*(range(size) for size in batch)))
    values = torch.arange(len(indices), device=device, dtype=dtype).reshape(batch)
    x = torch.stack((0.2 + 0.1 * values, -0.3 + 0.05 * values), dim=-1)
    weight = torch.stack((1 + 0.1 * values, 0.7 + 0.05 * values), dim=-1)
    phase = torch.stack((0.01 * values, -0.02 * values), dim=-1)
    u = torch.tensor([0.03], device=device, dtype=dtype) if shared else 0.03 + values[..., None] / 100
    scale = torch.ones((), device=device, dtype=dtype) if shared else 1 + values / 10
    P = scale[..., None, None] * torch.tensor([[0.5, 0.1], [0.1, 0.4]], device=device, dtype=dtype)
    Q = scale[..., None, None] * torch.eye(2, device=device, dtype=dtype) * 0.01
    R = scale[..., None, None] * torch.ones(1, 1, device=device, dtype=dtype) * 0.1
    model = NonlinearModel(weight, phase)
    batched = pp.module.EKF(model)
    individual = [pp.module.EKF(NonlinearModel(weight[index], phase[index])) for index in indices]
    expected_x = x.clone()
    expected_P = P.expand(batch + (2, 2)).clone()
    batch_u = u.expand(batch + (1,))
    batch_Q, batch_R = Q.expand(batch + (2, 2)), R.expand(batch + (1, 1))
    for step in range(3):
        time = torch.tensor(step, device=device) if shared else values + step
        batch_time = time.expand(batch)
        measurement = 0.2 + values[..., None] * 0.1 + step * 0.05
        x, P = batched(x, measurement, u, P, Q, R, t=time)
        for index, filter in zip(indices, individual):
            expected_x[index], expected_P[index] = filter(
                expected_x[index], measurement[index], batch_u[index], expected_P[index],
                batch_Q[index], batch_R[index], t=batch_time[index])
        torch.testing.assert_close(x, expected_x)
        torch.testing.assert_close(P, expected_P)
        assert x.shape == batch + (2,)
        assert P.shape == batch + (2, 2)


@pytest.mark.parametrize('name', ['A', 'C'])
@pytest.mark.parametrize('strategy', ['reverse-mode', 'forward-mode'])
def test_ekf_automatic_jacobians_match_independent(name, strategy):
    x = torch.linspace(0.1, 1.2, 12, dtype=torch.float64).reshape(2, 3, 2)
    u = torch.tensor([0.2], dtype=x.dtype)
    model = NonlinearModel(torch.ones_like(x), torch.zeros_like(x))
    model.jacargs['strategy'] = strategy
    time = torch.tensor(2)
    model.set_refpoint(x, u, t=time)
    actual = pp.module.EKF(model)._jacobian(name)
    expected = []
    for index in product(range(2), range(3)):
        single = NonlinearModel(model.weight[index].detach(), model.phase[index])
        single.jacargs['strategy'] = strategy
        single.set_refpoint(x[index], u, t=time)
        expected.append(getattr(single, name))
    torch.testing.assert_close(actual, torch.stack(expected).reshape(actual.shape))
    assert actual.shape == (2, 3, 2 if name == 'A' else 1, 2)


def test_ekf_preserves_analytic_overrides():
    class Analytic(NonlinearModel):
        @property
        def A(self):
            self.a_called = True
            return torch.diag_embed(1 + 0.1 * self.weight *
                                   (self.weight * self._ref_state + self.phase).cos())

        @property
        def C(self):
            self.c_called = True
            x = self._ref_state
            return torch.stack((2 * x[..., 0], 0.3 * x[..., 1].cos()), -1).unsqueeze(-2)

    x = torch.tensor([[0.2, 0.4], [0.1, -0.3]], dtype=torch.float64)
    weight, phase = torch.ones_like(x), torch.zeros_like(x)
    model = Analytic(weight, phase)
    args = (x, x[:, :1], torch.tensor([0.1], dtype=x.dtype),
            torch.eye(2, dtype=x.dtype), torch.eye(2, dtype=x.dtype) * 0.1,
            torch.ones(1, 1, dtype=x.dtype) * 0.2)
    actual = pp.module.EKF(model)(*args)
    expected = pp.module.EKF(NonlinearModel(weight, phase))(*args)
    assert model.a_called and model.c_called
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)


def test_ekf_preserves_linear_jacobians():
    dtype = torch.float64
    A, B = torch.eye(2, dtype=dtype), torch.ones(2, 1, dtype=dtype)
    C, D = torch.ones(1, 2, dtype=dtype), torch.zeros(1, 1, dtype=dtype)
    model = pp.module.LTI(A, B, C, D)
    filter = pp.module.EKF(model)
    assert filter._jacobian('A') is A
    assert filter._jacobian('C') is C


def test_general_nls_jacobians_keep_coupled_batch_semantics():
    class Coupled(pp.module.NLS):
        def state_transition(self, state, input, t=None):
            return state + state.sum(dim=0) + input

        def observation(self, state, input, t=None):
            return 2 * state + state.sum(dim=0) + input

    x = torch.randn(3, 2, dtype=torch.float64)
    u = torch.randn_like(x)
    model = Coupled().set_refpoint(x, u)
    for name, function, argument in [('A', model.state_transition, x),
                                     ('B', model.state_transition, u),
                                     ('C', model.observation, x),
                                     ('D', model.observation, u)]:
        func = (lambda z: function(z, u)) if name in ('A', 'C') else (lambda z: function(x, z))
        expected = torch.autograd.functional.jacobian(func, argument)
        torch.testing.assert_close(getattr(model, name), expected)
        assert expected.shape == (3, 2, 3, 2)


@pytest.mark.parametrize('create_graph', [False, True])
def test_ekf_batch_preserves_differentiation(create_graph):
    def run(batched):
        x = torch.tensor([[0.2, 0.4], [0.1, -0.3]], dtype=torch.float64, requires_grad=True)
        u = torch.tensor([0.1], dtype=x.dtype, requires_grad=True)
        model = NonlinearModel(torch.tensor([0.7, 0.9], dtype=x.dtype),
                               torch.tensor([0.02, -0.01], dtype=x.dtype))
        model.jacargs['create_graph'] = create_graph
        filter = pp.module.EKF(model)
        P, Q, R = torch.eye(2, dtype=x.dtype), torch.eye(2, dtype=x.dtype) * 0.1, \
            torch.ones(1, 1, dtype=x.dtype) * 0.2
        y = torch.ones(2, 1, dtype=x.dtype) * 0.3
        if batched:
            estimate, covariance = filter(x, y, u, P, Q, R)
            loss = estimate.square().sum() + covariance.square().sum()
        else:
            loss = 0
            for i in range(2):
                estimate, covariance = filter(x[i], y[i], u, P, Q, R)
                loss = loss + estimate.square().sum() + covariance.square().sum()
        return torch.autograd.grad(loss, (x, u, model.weight))

    for actual, expected in zip(run(True), run(False)):
        torch.testing.assert_close(actual, expected)
