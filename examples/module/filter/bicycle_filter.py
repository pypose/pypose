from pypose.module import EKF, PF
import torch, pypose as pp
import argparse, numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.legend_handler import HandlerLine2D


class Bicycle(pp.module.System):
    '''
    This is an implementation of the 2D Bicycle kinematic model,
    see: https://dingyan89.medium.com/simple-understanding-of-kinematic-bicycle-model-81cac6420357
    The robot is given a rotational and forward velocity, and traverses the 2D plane accordingly.
    This model is Nonlinear Time Invariant (NTI) and can be filtered with the ``pp.module.EKF``.
    '''

    def __init__(self):
        super().__init__()

    def state_transition(self, state, input, t=None):
        '''
        Don't add noise in this function, as it will be used for automatically
        linearizing the system by the parent class ``pp.module.System``.
        '''
        if state.ndim < 2:
            theta = state[2] + input[1]  # update heading (theta) from rotational velocity
            vx = input[0] * theta.cos()  # input[0] is magnitude of forward velocity
            vy = input[0] * theta.sin()
            return torch.stack([state[0] + vx, state[1] + vy, theta])

        else:
            theta = state[:, 2] + input[1]
            vx = input[0] * theta.cos()
            vy = input[0] * theta.sin()
            return torch.stack([state[:, 0] + vx, state[:, 1] + vy, theta], dim=1)

    def observation(self, state, input, t=None):
        '''
        Don't add noise in this function, as it will be used for automatically
        linearizing the system by the parent class ``pp.module.System``.
        '''
        return state


def createPlot(input, filter_name):
    fig, ax = plt.subplots()
    for idx, data in enumerate(input):
        state, est, cov = data
        name = filter_name[idx]
        N = state.shape[0]
        state = state.cpu().numpy()
        est = est.cpu().numpy()
        cov = cov.cpu().numpy()
        w = torch.arange(0, N, dtype=torch.float).view(-1, 1) / N
        c = torch.tensor([[1, 0, 0, 1]]).repeat(N, 1) * w + \
            torch.tensor([[0, 0, 1, 1]]).repeat(N, 1) * (1 - w)
        color = c.tolist()

        for i in range(N):
            eigvals, eigvecs = np.linalg.eig(cov[i])
            axis = np.sqrt(eigvals) * 3
            slope = eigvecs[1][0] / eigvecs[1][1]
            angle = 180.0 * np.arctan(slope) / np.pi
            e = Ellipse(est[i, 0:2], axis[0], axis[1], angle=angle)
            ax.add_artist(e)
            e.set_facecolor("none")
            e.set_edgecolor(color[i])
        if idx == 0:
            state_plot = ax.quiver(state[:-1, 0], state[:-1, 1],
                                   state[1:, 0] - state[:-1, 0], state[1:, 1] - state[:-1, 1],
                                   scale_units="xy", angles="xy", scale=1, color=color,
                                   label="True State")
        est_plot, = ax.plot(est[:, 0], est[:, 1], '.-', label="%s Estimated State" % name)
        ax.legend(handler_map={est_plot: HandlerLine2D(numpoints=1)})

    plt.title("Example")
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='EKF Example')
    parser.add_argument("--device", type=str, default='cpu', help="cuda or cpu")
    args = parser.parse_args()
    device = args.device
    T, N, M = 30, 3, 2  # steps, state dim, input dim
    q, r, p = 0.2, 0.2, 5  # covariance of transition noise, observation noise, and estimation

    input = torch.randn(T, M, device=device) * 0.1 + torch.tensor([1, 0], device=device)
    state = torch.zeros(T, N, device=device)  # true states
    est = torch.randn(T, N, device=device) * p  # estimation
    obs = torch.zeros(T, N, device=device)  # observation

    P = torch.eye(N, device=device).repeat(T, 1, 1) * p ** 2  # estimation covariance
    Q = torch.eye(N, device=device) * q ** 2  # covariance of transition
    R = torch.eye(N, device=device) * r ** 2  # covariance of observation

    # ekf
    input_ekf = input.clone()
    state_ekf = torch.zeros(T, N, device=device)  # true states
    est_ekf = est.clone()  # estimation
    obs_ekf = torch.zeros(T, N, device=device)  # observation
    P_ekf = P.clone()  # estimation covariance

    # robot
    robot_pf = Bicycle().to(device)
    robot_ekf = Bicycle().to(device)

    # filter
    ekf = EKF(robot_ekf, Q, R).to(device)
    pf = PF(robot_pf, Q, R, particle_number=1000).to(device)

    for i in range(T - 1):
        w = q * torch.randn(N, device=device)
        v = r * torch.randn(N, device=device)
        state[i + 1], obs[i] = robot_pf(state[i] + w, input[i])  # model measurement
        est[i + 1], P[i + 1] = pf(est[i], obs[i] + v, input[i], P[i])

        state_ekf[i + 1], obs_ekf[i] = robot_ekf(state_ekf[i] + w,
                                                 input_ekf[i])  # model measurement
        est_ekf[i + 1], P_ekf[i + 1] = ekf(est_ekf[i], obs_ekf[i] + v, input_ekf[i], P_ekf[i])

    print('pf', (state - est).norm(dim=-1).sum())
    print('ekf', (state - est_ekf).norm(dim=-1).sum())
    createPlot([[state, est, P], [state_ekf, est_ekf, P_ekf]],
               ['pf', 'ekf'])
