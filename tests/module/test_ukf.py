import pypose as pp
import torch as torch
import pytest
from itertools import product


class TestUKF:

    def test_sigma_points_preserve_correlated_covariance(self):
        ukf = pp.module.UKF(pp.module.NLS())
        x = torch.tensor([1., 2.], dtype=torch.float64)
        P = torch.tensor([[2., 0.5], [0.5, 1.]], dtype=torch.float64)

        xs, w = ukf.sigma_weight_points(x, P, k=1)
        mean = (w * xs).sum(dim=-2)
        covariance = ukf.compute_cov(xs - mean, xs - mean, w)

        assert torch.allclose(mean, x)
        assert torch.allclose(covariance, P)

    def test_update_matches_linear_kalman_filter_with_process_noise(self):
        class NLS(pp.module.NLS):
            def state_transition(self, state, input, t=None):
                return state + input

            def observation(self, state, input, t=None):
                return state

        ukf = pp.module.UKF(NLS())
        x = torch.tensor([0.], dtype=torch.float64)
        y = torch.tensor([2.], dtype=torch.float64)
        u = torch.tensor([1.], dtype=torch.float64)
        P = torch.tensor([[1.]], dtype=torch.float64)
        Q = torch.tensor([[3.]], dtype=torch.float64)
        R = torch.tensor([[1.]], dtype=torch.float64)

        xp, Pp = ukf(x, y, u, P, Q, R)

        # Predicted covariance is 4, so the Kalman gain is 4 / (4 + 1).
        assert torch.allclose(xp, torch.tensor([1.8], dtype=torch.float64))
        assert torch.allclose(Pp, torch.tensor([[0.8]], dtype=torch.float64))

    @pytest.mark.parametrize("batch", [(), (3,), (2, 3)])
    @pytest.mark.parametrize("shared_covariance", [False, True])
    def test_batched_linear_update(self, batch, shared_covariance):
        class NLS(pp.module.NLS):
            def state_transition(self, state, input, t=None):
                return state + input

            def observation(self, state, input, t=None):
                return state

        ukf = pp.module.UKF(NLS())
        x = torch.arange(2 * torch.Size(batch).numel(), dtype=torch.float64).reshape(*batch, 2)
        P = torch.tensor([[2., 0.5], [0.5, 1.]], dtype=x.dtype)
        if not shared_covariance:
            P = P.expand(*batch, 2, 2)
        Q, R = torch.eye(2, dtype=x.dtype) * 0.3, torch.eye(2, dtype=x.dtype) * 0.5
        xs, w = ukf.sigma_weight_points(x, P, k=1)
        mean = (w * xs).sum(dim=0)
        torch.testing.assert_close(mean, x)
        torch.testing.assert_close(ukf.compute_cov(xs - mean, xs - mean, w),
                                   P.expand(*batch, 2, 2))
        actual_x, actual_P = ukf(x, x + 2, torch.ones_like(x), P, Q, R)
        gain = (P + Q) @ torch.linalg.inv(P + Q + R)
        expected_x = x + 1 + (gain @ torch.ones_like(x).unsqueeze(-1)).squeeze(-1)
        expected_P = P + Q - gain @ (P + Q + R) @ gain.mT
        torch.testing.assert_close(actual_x, expected_x)
        torch.testing.assert_close(actual_P, expected_P.expand(*batch, 2, 2))

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

        batched = pp.module.UKF(NLS())
        independent = [pp.module.UKF(NLS()) for _ in indices]
        for step in range(3):
            measurement = y + 0.1 * step
            x, P = batched(x, measurement, u, P, Q, R)
            for index, filter in zip(indices, independent):
                expected_x[index], expected_P[index] = filter(
                    expected_x[index], measurement[index], u[index],
                    expected_P[index], batch_Q[index], batch_R[index])
            torch.testing.assert_close(x, expected_x)
            torch.testing.assert_close(P, expected_P)

    def test_ukf(self):
        class NLS(pp.module.NLS):
            def __init__(self):
                super().__init__()

            def forward(self, state, input, noise):
                state = self.state_transition(state, input, self.systime) + noise
                return state, self.observation(state, input, self.systime + 1)

            def state_transition(self, state, input, t=None):
                return state.cos() + input

            def observation(self, state, input, t=None):
                return state.sin() + input

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = NLS().to(device)
        ukf = pp.module.UKF(model).to(device)

        T, N = 5, 2  # steps, state dim
        states = torch.zeros(T, N, device=device)
        inputs = torch.randn(T, N, device=device)
        observ = torch.zeros(T, N, device=device)
        # std of transition, observation, and estimation
        q, r, p = 0.1, 0.1, 10
        Q = torch.eye(N, device=device) * q ** 2
        R = torch.eye(N, device=device) * r ** 2
        P = torch.eye(N, device=device).repeat(T, 1, 1) * p ** 2
        estim = torch.ones(T, N, device=device) * p

        for i in range(T - 1):
            w = q * torch.randn(N, device=device)  # transition noise
            v = r * torch.randn(N, device=device)  # observation noise
            states[i + 1], observ[i + 1] = model(states[i], inputs[i], w)
            estim[i + 1], P[i + 1] = ukf(estim[i], observ[i + 1] + v, inputs[i], P[i], Q, R)
        error = (states - estim).norm(dim=-1)

        assert torch.all(error[0] - error[-1] > 0), "Filter error last step too large."


if __name__ == '__main__':
    test = TestUKF()
    test.test_ukf()
