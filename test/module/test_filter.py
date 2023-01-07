import pypose as pp
import torch as torch


class TestFilter:

    def test_filter(self):
        class NTI(pp.module.System):
            def __init__(self):
                super().__init__()

            def state_transition(self, state, input, t=None):
                return state.cos() + input

            def observation(self, state, input, t=None):
                return state.sin() + input

        device = 'cpu'
        model_ekf = NTI().to(device)
        model_pf = NTI().to(device)

        ekf = pp.module.EKF(model_ekf).to(device)
        pf = pp.module.PF(model_pf, particle_number=1000).to(device)

        T, N = 10, 3  # steps, state dim
        states = torch.zeros(T, N, device=device)
        inputs = torch.randn(T, N, device=device)
        observ = torch.zeros(T, N, device=device)
        # std of transition, observation, and estimation
        q, r, p = 0.1, 0.1, 10
        Q = torch.eye(N, device=device) * q ** 2
        R = torch.eye(N, device=device) * r ** 2
        P = torch.eye(N, device=device).repeat(T, 1, 1) * p ** 2
        estim = torch.randn(T, N, device=device) * p

        # pf
        inputs_pf = inputs.clone()
        states_pf = torch.zeros(T, N, device=device)
        observ_pf = torch.zeros(T, N, device=device)
        estim_pf = estim.clone()
        P_pf = P.clone()

        for i in range(T - 1):
            w = q * torch.randn(N, device=device)  # transition noise
            v = r * torch.randn(N, device=device)  # observation noise
            states[i + 1], observ[i] = model_ekf(states[i] + w, inputs[i])
            estim[i + 1], P[i + 1] = ekf(estim[i], observ[i] + v, inputs[i], P[i], Q, R)

            states_pf[i + 1], observ_pf[i] = model_pf(states_pf[i] + w, inputs_pf[i])
            estim_pf[i + 1], P_pf[i + 1] = pf(estim_pf[i], observ_pf[i] + v, inputs_pf[i],
                                              P_pf[i], Q, R)

        error_ekf = (states - estim).norm(dim=-1)
        error_pf = (states_pf - estim_pf).norm(dim=-1)
        print('ekf', error_ekf.mean())
        print('pf', error_pf.mean())


if __name__ == '__main__':
    test = TestFilter()
    test.test_filter()
