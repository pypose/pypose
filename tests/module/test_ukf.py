import pypose as pp
import torch as torch


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

    def test_ukf(self):
        class NLS(pp.module.NLS):
            def __init__(self):
                super().__init__()

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
            states[i + 1], _ = model(states[i] + w, inputs[i])
            observ[i + 1] = model.observation(states[i + 1], inputs[i])
            estim[i + 1], P[i + 1] = ukf(estim[i], observ[i + 1] + v, inputs[i], P[i], Q, R)
        error = (states - estim).norm(dim=-1)

        assert torch.all(error[0] - error[-1] > 0), "Filter error last step too large."


if __name__ == '__main__':
    test = TestUKF()
    test.test_ukf()
