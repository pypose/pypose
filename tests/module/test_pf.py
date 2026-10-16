import pypose as pp
import torch as torch
import pytest


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
