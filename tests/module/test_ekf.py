import math
import pypose as pp
import pytest
import torch as torch


class TestEKF:

    def test_ekf(self):
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
        ekf = pp.module.EKF(model).to(device)

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
            estim[i + 1], P[i + 1] = ekf(estim[i], observ[i + 1] + v, inputs[i], P[i], Q, R)
        error = (states - estim).norm(dim=-1)

        assert torch.all(error[0] - error[-1] > 0), "Filter error last step too large."

    def test_update_linearizes_at_predicted_state(self):
        class NLS(pp.module.NLS):
            def state_transition(self, state, input, t=None):
                return 2 * state

            def observation(self, state, input, t=None):
                return state.square()

        model = NLS()
        ekf = pp.module.EKF(model)
        x = torch.tensor([1.], dtype=torch.float64)
        y = torch.tensor([5.], dtype=torch.float64)
        u = torch.tensor([0.], dtype=torch.float64)
        P = torch.tensor([[1.]], dtype=torch.float64)
        Q = torch.tensor([[0.]], dtype=torch.float64)
        R = torch.tensor([[1.]], dtype=torch.float64)

        xp, Pp = ekf(x, y, u, P, Q, R)

        assert torch.allclose(xp, torch.tensor([2 + 16 / 65], dtype=torch.float64))
        assert torch.allclose(Pp, torch.tensor([[4 / 65]], dtype=torch.float64))

    @pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
    def test_small_measurement_noise_does_not_freeze_later_updates(self, dtype):
        class Identity(pp.module.NLS):
            def state_transition(self, state, input, t=None):
                return state

            def observation(self, state, input, t=None):
                return state

        ekf = pp.module.EKF(Identity(), Q=torch.zeros(1, 1, dtype=dtype),
                           R=torch.tensor([[0.01]], dtype=dtype))
        state = torch.zeros(1, dtype=dtype)
        covariance = torch.tensor([[1e6]], dtype=dtype)
        precision = 1e-6
        information = 0.0
        for measurement in (1.0, 2.0, -1.0):
            state, covariance = ekf(state, torch.tensor([measurement], dtype=dtype),
                                    torch.zeros_like(state), covariance)
            precision += 100.0
            information += 100.0 * measurement
            torch.testing.assert_close(covariance, torch.tensor([[1 / precision]], dtype=dtype))
            torch.testing.assert_close(state, torch.tensor([information / precision], dtype=dtype))

    @pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
    @pytest.mark.parametrize('batch_shape', [(), (2,)])
    def test_correlated_covariance_matches_information_form(self, dtype, batch_shape):
        class Identity(pp.module.NLS):
            def state_transition(self, state, input, t=None):
                return state

            def observation(self, state, input, t=None):
                return state

            @property
            def A(self):
                return torch.eye(2, dtype=dtype)

            @property
            def C(self):
                return torch.eye(2, dtype=dtype)

        covariance = 1e6 * torch.tensor([[2.0, 0.3], [0.3, 1.0]], dtype=dtype)
        covariance = covariance.expand(batch_shape + (2, 2)).clone()
        noise = torch.tensor([[0.04, 0.01], [0.01, 0.09]], dtype=dtype)
        observation = torch.tensor([1.0, -2.0], dtype=dtype).expand(batch_shape + (2,))
        prior = torch.zeros_like(observation)
        ekf = pp.module.EKF(Identity(), Q=torch.zeros(2, 2, dtype=dtype), R=noise)

        state, actual = ekf(prior, observation, torch.zeros_like(prior), covariance)

        prior_precision = torch.linalg.inv(covariance.double())
        noise_precision = torch.linalg.inv(noise.double())
        expected = torch.linalg.inv(prior_precision + noise_precision)
        expected_state = (expected @ noise_precision @ observation.double().unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(actual, expected.to(dtype))
        torch.testing.assert_close(state, expected_state.to(dtype))
        torch.testing.assert_close(actual, actual.mT)
        assert torch.all(torch.linalg.eigvalsh(actual) > 0)


if __name__ == '__main__':
    test = TestEKF()
    test.test_ekf()
