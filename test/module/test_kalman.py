import math
import pypose as pp
import torch as torch


class TestKalman:

    def test_kalman(self):

        class NTI(pp.module.System):
            def __init__(self):
                super().__init__()

            def state_transition(self, state, input, t=None):
                return state.cos() + input

            def observation(self, state, input, t):
                return state.sin() + input

        model = NTI()
        ekf = pp.module.EKF(model)

        T, N = 5, 2 # steps, state dim
        states = torch.zeros(T, N)
        inputs = torch.randn(T, N)
        observ = torch.zeros(T, N)
        # std of transition, observation, and estimation
        q, r, p = 0.1, 0.1, 10
        Q = torch.eye(N) * q**2
        R = torch.eye(N) * r**2
        P = torch.eye(N).repeat(T, 1, 1) * p**2
        estim = torch.randn(T, N) * p

        for i in range(T - 1):
            w = q * torch.randn(N) # transition noise
            v = r * torch.randn(N) # observation noise
            states[i+1], observ[i] = model(states[i] + w, inputs[i])
            estim[i+1], P[i+1] = ekf(estim[i], observ[i] + v, inputs[i], P[i], Q, R)
        error =  (states - estim).norm(dim=-1)

        assert torch.all(error[0] - error[-1] > 0), "Filter error last step too large."


if __name__ == '__main__':
    test = TestKalman()
    test.test_kalman()
