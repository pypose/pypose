import pypose as pp
import torch as torch
import pytest
from itertools import product
from unittest.mock import patch
from contextlib import contextmanager


class TestPF:

    @pytest.mark.parametrize("batch", [(), (3,), (2, 3)])
    def test_particle_batch_axes(self, batch):
        pf = pp.module.PF(pp.module.NLS(), particles=5)
        x = torch.zeros(*batch, 2, dtype=torch.float64)
        ye = torch.arange(10., dtype=x.dtype).reshape(5, *([1] * len(batch)), 2)
        ye = ye.expand(5, *batch, 2)
        R = torch.eye(2, dtype=x.dtype)
        q = pf.relative_likelihood(x, ye, R)
        expected = torch.softmax(torch.distributions.MultivariateNormal(ye, R).log_prob(x),
                                 dim=0)
        torch.testing.assert_close(q, expected)
        torch.testing.assert_close(q.sum(dim=0), torch.ones(batch, dtype=x.dtype))
        # A point mass must select the same particle independently in every batch.
        selected = torch.arange(torch.Size(batch).numel()).reshape(batch) % 5
        q = torch.zeros_like(q).scatter_(0, selected.unsqueeze(0), 1)
        actual = pf.resample_particles(q, ye)
        expected = torch.gather(ye, 0, selected.unsqueeze(0).unsqueeze(-1)
                                .expand(5, *batch, 2))
        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize("batch", [(3,), (2, 3)])
    @pytest.mark.parametrize("shared_covariance", [False, True])
    def test_batched_matches_independent_updates(self, batch, shared_covariance):
        class NLS(pp.module.NLS):
            def state_transition(self, state, input, t=None):
                return state + 0.1 * state.sin() + input

            def observation(self, state, input, t=None):
                return state + 0.2 * state.sin()

        indices = list(product(*(range(size) for size in batch)))
        values = torch.arange(len(indices), dtype=torch.float64).reshape(batch)
        x = torch.stack((0.2 * values, -0.3 * values), dim=-1)
        u = torch.stack((0.03 * values + 0.1, 0.02 * values - 0.1), dim=-1)
        y = x + torch.stack((0.05 * values + 0.3, -0.04 * values + 0.2), dim=-1)

        base = torch.tensor([[0.8, 0.15], [0.15, 0.5]], dtype=x.dtype)
        scale = 0 if shared_covariance else values[..., None, None]
        P = (1 + 0.1 * scale) * base
        Q = (0.1 + 0.01 * scale) * base
        R = (0.2 + 0.02 * scale) * base
        expected_x = x.clone()
        expected_P = P.expand(*batch, 2, 2).clone()
        batch_Q, batch_R = Q.expand(*batch, 2, 2), R.expand(*batch, 2, 2)

        @contextmanager
        def fixed_draws(normals, uniforms):
            # Control only randomness; exercise the actual particle-filter operations.
            with patch("torch.distributions.multivariate_normal._standard_normal",
                       return_value=normals), patch("torch.rand", return_value=uniforms):
                yield

        particles = 64
        batched = pp.module.PF(NLS(), particles=particles)
        independent = [pp.module.PF(NLS(), particles=particles) for _ in indices]
        generator = torch.Generator().manual_seed(0)
        for step in range(3):
            measurement = y + 0.1 * step
            normals = torch.randn(particles, *batch, 2, dtype=x.dtype, generator=generator)
            uniforms = torch.rand(particles, *batch, dtype=x.dtype, generator=generator)
            with fixed_draws(normals, uniforms):
                x, P = batched(x, measurement, u, P, Q, R)
            for index, filter in zip(indices, independent):
                draws = (slice(None),) + index
                with fixed_draws(normals[draws], uniforms[draws]):
                    expected_x[index], expected_P[index] = filter(
                        expected_x[index], measurement[index], u[index],
                        expected_P[index], batch_Q[index], batch_R[index])
            torch.testing.assert_close(x, expected_x)
            torch.testing.assert_close(P, expected_P)

    def test_pf(self):
        class NLS(pp.module.NLS):
            def __init__(self):
                super().__init__()

            def state_transition(self, state, input, t=None):
                return state.cos() + input

            def observation(self, state, input, t=None):
                return state.sin() + input

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = NLS().to(device)
        pf = pp.module.PF(model).to(device)

        B, T, N = 3, 5, 2  # steps, state dim
        states = torch.zeros(B, T, N, device=device)
        inputs = torch.randn(B, T, N, device=device)
        observ = torch.zeros(B, T, N, device=device)
        # std of transition, observation, and estimation
        q, r, p = 0.1, 0.1, 10
        Q = torch.eye(N, device=device).repeat(B, 1, 1) * q ** 2
        R = torch.eye(N, device=device).repeat(B, 1, 1) * r ** 2
        P = torch.eye(N, device=device).repeat(B, T, 1, 1) * p ** 2
        estim = torch.ones(B, T, N, device=device) * p

        for i in range(T - 1):
            w = q * torch.randn(B, N, device=device)  # transition noise
            v = r * torch.randn(B, N, device=device)  # observation noise
            states[:, i+1], observ[:, i] = model(states[:, i] + w, inputs[:, i])
            estim[:, i+1], P[:, i+1] = pf(estim[:, i], observ[:, i] + v, inputs[:, i], P[:, i], Q, R)
        error = (states - estim).norm(dim=-1)

        assert torch.all(error[:, 0] - error[:, -1] > 0), "Filter error last step too large."


if __name__ == '__main__':
    test = TestPF()
    test.test_pf()
