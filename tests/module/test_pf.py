import pypose as pp
import torch as torch


class TestPF:

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
        estim = torch.randn(B, T, N, device=device) * p
        for i in range(T - 1):
            w = q * torch.randn(B, N, device=device)  # transition noise
            v = r * torch.randn(B, N, device=device)  # observation noise
            states[:, i+1], observ[:, i] = model(states[:, i] + w, inputs[:, i])
            estim[:, i+1], P[:, i+1] = pf(estim[:, i], observ[:, i] + v, inputs[:, i], P[:, i], Q, R)
        error = (states - estim).norm(dim=-1)

        assert torch.all(error[0] - error[-1] > 0), "Filter error last step too large."


if __name__ == '__main__':
    test = TestPF()
    test.test_pf()
