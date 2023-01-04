import torch
import pypose as pp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.legend_handler import HandlerLine2D
from pypose.module import EKF, UKF


def model_factory(robot, model_name, device, Q, R):
    r"""
    filter model factory

    Args:
        robot: real machine model
        model_name(:obj:`Str`, optional):name of estimator
        device(:obj:`Str`, optional):device of pytorch
        Q(:obj:`Tensor`):covariance of transition
        R(:obj:`Tensor`):covariance of observation

    Return:
        FilterModel(:obj:):filter model of robot
    """

    if model_name.lower() == 'ekf':
        return EKF(robot, Q, R).to(device)

    elif model_name.lower() == 'ukf':
        return UKF(robot, Q, R).to(device)

    else:
        raise ValueError('The model_name parameter is incorrect')


class Bicycle(pp.module.System):
    '''
    This is an implementation of the 2D Bicycle kinematic model,
    see: https://dingyan89.medium.com/simple-understanding-of-kinematic-bicycle-model
    -81cac6420357
    The robot is given a rotational and forward velocity, and traverses the 2D plane accordingly.
    This model is Nonlinear Time Invariant (NTI) and can be filtered with the ``pp.module.EKF``
    and  ``pp.module.UKF``.

    Args:
        T(:obj:`Int`, optional):steps
        N(:obj:`Int`, optional):state dim
        M(:obj:`Int`, optional):input dim
        q(:obj:`Float`, optional):covariance of transition noise
        r(:obj:`Float`, optional):observation noise
        p(:obj:`Int`, optional):estimation
        model_name(:obj:`Str`, optional):name of estimator
        device(:obj:`Str`, optional):device of pytorch
    '''

    def __init__(self, T=30, N=3, M=2, q=0.2, r=0.2, p=5, model_name='ekf', device='cpu'):
        super().__init__()
        self.T = T
        self.N = N
        self.M = M
        self.q = q
        self.r = r
        self.p = p
        self.model_name = model_name
        self.device = device

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

    def createPlot(self, state, est, cov):
        N = state.shape[0]
        state = state.cpu().numpy()
        est = est.cpu().numpy()
        cov = cov.cpu().numpy()
        w = torch.arange(0, N, dtype=torch.float).view(-1, 1) / N
        c = torch.tensor([[1, 0, 0, 1]]).repeat(N, 1) * w + \
            torch.tensor([[0, 0, 1, 1]]).repeat(N, 1) * (1 - w)
        color = c.tolist()
        fig, ax = plt.subplots()
        for i in range(N):
            eigvals, eigvecs = np.linalg.eig(cov[i])
            axis = np.sqrt(eigvals) * 3
            slope = eigvecs[1][0] / eigvecs[1][1]
            angle = 180.0 * np.arctan(slope) / np.pi
            e = Ellipse(est[i, 0:2], axis[0], axis[1], angle=angle)
            ax.add_artist(e)
            e.set_facecolor("none")
            e.set_edgecolor(color[i])
        state_plot = ax.quiver(state[:-1, 0], state[:-1, 1],
                               state[1:, 0] - state[:-1, 0], state[1:, 1] - state[:-1, 1],
                               scale_units="xy", angles="xy", scale=1, color=color,
                               label="True State")
        est_plot, = ax.plot(est[:, 0], est[:, 1], '.-', label="Estimated State")
        ax.legend(handler_map={est_plot: HandlerLine2D(numpoints=1)})
        plt.title("%s Example" % self.model_name.upper())
        plt.show()

    def run_estimate(self, robot):
        r"""
        run estimate task
        """
        assert robot is not None, 'bicycle robot is uninitialized'

        input = torch.randn(self.T, self.M, device=self.device) * 0.1 + \
                torch.tensor([1, 0], device=self.device)
        state = torch.zeros(self.T, self.N, device=self.device)  # true states
        est = torch.randn(self.T, self.N, device=self.device) * self.p  # estimation
        obs = torch.zeros(self.T, self.N, device=self.device)  # observation
        P = torch.eye(self.N, device=self.device).repeat(
            self.T, 1, 1) * self.p ** 2  # estimation
        # covariance
        Q = torch.eye(self.N, device=self.device) * self.q ** 2  # covariance of transition
        R = torch.eye(self.N, device=self.device) * self.r ** 2  # covariance of observation

        filter_model = model_factory(robot, self.model_name, self.device, Q, R)

        for i in range(self.T - 1):
            w = self.q * torch.randn(self.N, device=self.device)
            v = self.r * torch.randn(self.N, device=self.device)
            state[i + 1], obs[i] = robot(state[i] + w, input[i])  # model measurement
            est[i + 1], P[i + 1] = filter_model(est[i], obs[i] + v, input[i], P[i])

        robot.createPlot(state, est, P)
