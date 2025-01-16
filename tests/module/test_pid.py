import math
import pypose as pp
import torch as torch


class TestPID:

    def test_pid(self):
        kp = 0.6
        ki = 0.1
        kd = 0.3
        pid = pp.module.PID(kp, ki, kd)

        N = 20
        dt = 0.1
        state = torch.zeros(N, 1)
        state_sp = torch.ones(N, 1)

        for i in range(N - 1):
            error = state_sp[i] - state[i]
            error_dot = (state_sp[i] - state[i]) / dt
            state[i+1] = state[i] + pid.forward(error, error_dot) * dt
            state[i+1] -= 0.05 # add stable residual error

        state_ref = torch.tensor([
            [0.0000],
            [0.3200],
            [0.5316],
            [0.6717],
            [0.7647],
            [0.8265],
            [0.8678],
            [0.8956],
            [0.9144],
            [0.9273],
            [0.9363],
            [0.9427],
            [0.9473],
            [0.9508],
            [0.9536],
            [0.9558],
            [0.9576],
            [0.9593],
            [0.9607],
            [0.9620]])

        torch.testing.assert_close(state_ref, state, atol=1e-5, rtol=1e-3)

if __name__ == "__main__":
    epnp_fixture = TestPID()
    epnp_fixture.test_pid()
