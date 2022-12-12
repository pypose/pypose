import argparse
from pypose.module import EKF
import math, matplotlib.pyplot as plt
import torch, pypose as pp, numpy as np


class TankRobot(pp.module.System):
    def __init__(self, Q, R):
        super(TankRobot, self).__init__()
        self.register_buffer("Q", Q)
        self.register_buffer("R", R)

    def state_transition(self, state, input, t=None):
        theta = state[2] + input[1]
        vx = input[0] * theta.cos()
        vy = input[0] * theta.sin()
        state = torch.stack([state[0] + vx, state[1] + vy, theta])
        return state + self.noise(self.Q)

    def observation(self, state, input, t=None):
        return state + self.noise(self.R)

    def noise(self, W):
        r'''
        Randomly generated batched noises.
        '''
        n = torch.randn(W.shape[:-1], device=W.device, dtype=W.dtype)
        return pp.bmv(W, n)


def creatPlot(state, est):
    N = state.shape[0]
    state = state.cpu().numpy()
    est = est.cpu().numpy()
    w = torch.arange(0, N, dtype=torch.float).view(-1, 1) / N
    c = torch.tensor([[1, 0, 0, 1]]).repeat(N, 1) * w + \
        torch.tensor([[0, 0, 1, 1]]).repeat(N, 1) * (1 - w)
    plt.quiver(state[:-1,0], state[:-1,1],
               state[1:,0]-state[:-1,0], state[1:,1]-state[:-1,1],
               scale_units="xy", angles="xy", scale=1, color=c.tolist())
    plt.plot(est[:, 0], est[:, 1], '.-')
    plt.legend(['True State', 'Estimated State'])
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='IMU Preintegration')
    parser.add_argument("--device", type=str, default='cpu', help="cuda or cpu")
    args = parser.parse_args()

    device = args.device
    N, q, r = 100, 0.2, 0.2  # steps, covariance of transition and observation noise
    input  = torch.randn(N, 2, device=device) * 0.1 + torch.tensor([1, 0], device=device)
    state  = torch.zeros(N, 3, device=device)               # true states
    est    = torch.randn(N, 3, device=device) + 10          # estimation
    obs    = torch.zeros(N, 3, device=device)               # observation
    cov    = torch.eye(3, 3, device=device).repeat(N, 1, 1) # estimation covariance
    Q      = torch.eye(3, device=device) * q                # covariance of transition
    R      = torch.eye(3, device=device) * r                # covariance of observation

    robot = TankRobot(Q, R).to(device)
    ekf = EKF(robot, Q, R).to(device)

    for i in range(N - 1):
        state[i+1], obs[i] = robot(state[i], input[i])  # model measurement
        est[i+1], cov[i+1] = ekf(est[i], obs[i], input[i], cov[i])

    creatPlot(state, est)
