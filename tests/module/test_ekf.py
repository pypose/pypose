import math
import pypose as pp
import torch as torch
import pytest


class TestEKF:

    @pytest.mark.parametrize('nonlinear', [False, True])
    @pytest.mark.parametrize('innovation', [0.0, 1.0])
    def test_update_uses_predicted_state(self, nonlinear, innovation):
        class Model(pp.module.NLS):
            def state_transition(self, state, input, t=None):
                return 2 * state + input

            def observation(self, state, input, t=None):
                return state.square() if nonlinear else 3 * state

        dtype = torch.float64
        x = torch.tensor([1.0], dtype=dtype)
        u = torch.tensor([1.0], dtype=dtype)
        P = torch.tensor([[1.0]], dtype=dtype)
        Q = torch.tensor([[0.1]], dtype=dtype)
        R = torch.tensor([[0.5]], dtype=dtype)
        # x^- = 3, P^- = 2^2 * 1 + 0.1. Both observation models give h(3) = 9.
        y = torch.tensor([9.0 + innovation], dtype=dtype)
        H = 6.0 if nonlinear else 3.0
        gain = 4.1 * H / (H * H * 4.1 + 0.5)
        expected_x = torch.tensor([3.0 + gain * innovation], dtype=dtype)
        expected_P = torch.tensor([[4.1 * 0.5 / (H * H * 4.1 + 0.5)]], dtype=dtype)

        posterior_x, posterior_P = pp.module.EKF(Model(), Q, R)(x, y, u, P)
        torch.testing.assert_close(posterior_P, expected_P)
        torch.testing.assert_close(posterior_x, expected_x)

    def test_ekf(self):
        class NLS(pp.module.NLS):
            def __init__(self):
                super().__init__()

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
            states[i + 1] = model.state_transition(states[i] + w, inputs[i])
            observ[i] = model.observation(states[i + 1], inputs[i])
            estim[i + 1], P[i + 1] = ekf(estim[i], observ[i] + v, inputs[i], P[i], Q, R)
        error = (states - estim).norm(dim=-1)

        assert torch.all(error[0] - error[-1] > 0), "Filter error last step too large."


if __name__ == '__main__':
    test = TestEKF()
    test.test_ekf()
