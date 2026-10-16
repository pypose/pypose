import pypose as pp
import torch as torch
import pytest
from unittest.mock import patch


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

        count = torch.Size(batch).numel()
        values = torch.arange(count, dtype=torch.float64)
        x = torch.stack((values * 0.2, values * -0.3), dim=-1)
        u = torch.stack((values * 0.03 + 0.1, values * 0.02 - 0.1), dim=-1)
        y = x + torch.stack((values * 0.05 + 0.3, values * -0.04 + 0.2), dim=-1)
        base = torch.tensor([[0.8, 0.15], [0.15, 0.5]], dtype=x.dtype)
        P = (1 + values[:, None, None] * 0.1) * base
        Q = (0.1 + values[:, None, None] * 0.01) * base
        R = (0.2 + values[:, None, None] * 0.02) * base
        if shared_covariance:
            P, Q, R = [matrix[0].expand_as(matrix) for matrix in (P, Q, R)]

        particles = 64
        batched = pp.module.PF(NLS(), particles=particles)
        independent = [pp.module.PF(NLS(), particles=particles) for _ in range(count)]
        generator = torch.Generator().manual_seed(0)
        batch_x, batch_P = x.reshape(*batch, 2), P.reshape(*batch, 2, 2)
        for step in range(3):
            # Supply identical standard-normal and uniform draws to both paths.
            # Keep the production sampling, likelihood, and resampling algorithms.
            normals = torch.randn(particles, count, 2, dtype=x.dtype, generator=generator)
            uniforms = torch.rand(particles, count, dtype=x.dtype, generator=generator)
            with patch("torch.distributions.multivariate_normal._standard_normal",
                       return_value=normals.reshape(particles, *batch, 2)), \
                 patch("torch.rand", return_value=uniforms.reshape(particles, *batch)):
                batch_x, batch_P = batched(
                    batch_x, (y + step * 0.1).reshape(*batch, 2), u.reshape(*batch, 2),
                    batch_P if step else (P[0] if shared_covariance else batch_P),
                    Q[0] if shared_covariance else Q.reshape(*batch, 2, 2),
                    R[0] if shared_covariance else R.reshape(*batch, 2, 2))
            results = []
            for i in range(count):
                with patch("torch.distributions.multivariate_normal._standard_normal",
                           return_value=normals[:, i]), \
                     patch("torch.rand", return_value=uniforms[:, i]):
                    results.append(independent[i](x[i], y[i] + step * 0.1,
                                                  u[i], P[i], Q[i], R[i]))
            x, P = [torch.stack(items) for items in zip(*results)]
            torch.testing.assert_close(batch_x, x.reshape(*batch, 2))
            torch.testing.assert_close(batch_P, P.reshape(*batch, 2, 2))

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
