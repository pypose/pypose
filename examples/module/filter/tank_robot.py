import argparse
import torch, pypose as pp
from pypose.module import EKF
import matplotlib.pyplot as plt


class TankRobot(pp.module.System):
    def __init__(self, Q, R):
        super(TankRobot, self).__init__()
        self.register_buffer("Q", Q)
        self.register_buffer("R", R)

    def state_transition(self, state, input, t=None):
        '''
        Don't add noise in this function, as it will be used for automatically
        linearizing the system by the parent class ``pp.module.System``.
        '''
        theta = state[2] + input[1]
        vx = input[0] * theta.cos()
        vy = input[0] * theta.sin()
        return torch.stack([state[0] + vx, state[1] + vy, theta])

    def observation(self, state, input, t=None):
        '''
        Don't add noise in this function, as it will be used for automatically
        linearizing the system by the parent class ``pp.module.System``.
        '''
        return state


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
    T, N, M = 100, 3, 2     # steps, state dim, input dim
    q, r, p = 0.2, 0.2, 10  # covariance of transition noise, observation noise, and estimation
    input = torch.randn(T, M, device=device) * 0.1 + torch.tensor([1, 0], device=device)
    state = torch.zeros(T, N, device=device)                   # true states
    est   = torch.randn(T, N, device=device) * p               # estimation
    obs   = torch.zeros(T, N, device=device)                   # observation
    P     = torch.eye(N, device=device).repeat(T, 1, 1) * p**2 # estimation covariance
    Q     = torch.eye(N, device=device) * q**2                 # covariance of transition
    R     = torch.eye(N, device=device) * r**2                 # covariance of observation

    robot = TankRobot(Q, R).to(device)
    ekf = EKF(robot, Q, R).to(device)

    for i in range(T - 1):
        w = q * torch.randn(N, device=device)
        v = r * torch.randn(N, device=device)
        state[i+1], obs[i] = robot(state[i] + w, input[i])  # model measurement
        est[i+1], P[i+1] = ekf(est[i], obs[i] + v, input[i], P[i])

    creatPlot(state, est)
