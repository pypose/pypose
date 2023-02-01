import argparse
from pypose.module import EKF, UKF
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.legend_handler import HandlerLine2D
from bicycle import Bicycle


def filter_model_factory(robot, model_name, device, Q, R):
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


def createPlot(model_name, state, est, cov):
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
    plt.title("%s Example" % model_name.upper())
    plt.show()


def run_estimate(robot, k, T, N, M, q, r, p, filter_model_name, device):
    r"""
    run estimate task

    Args:
        T(:obj:`Int`, optional):steps
        N(:obj:`Int`, optional):state dim
        M(:obj:`Int`, optional):input dim
        q(:obj:`Float`, optional):covariance of transition noise
        r(:obj:`Float`, optional):observation noise
        p(:obj:`Int`, optional):estimation
        filter_model_name(:obj:`Str`, optional):name of filter model
        device(:obj:`Str`, optional):device of pytorch
    """
    assert robot is not None, 'bicycle robot is uninitialized'
    input = torch.randn(T, M, device=device) * 0.1 + \
            torch.tensor([1, 0], device=device)
    state = torch.zeros(T, N, device=device)  # true states
    est = torch.randn(T, N, device=device) * p  # estimation
    obs = torch.zeros(T, N, device=device)  # observation
    P = torch.eye(N, device=device).repeat(
        T, 1, 1) * p ** 2  # estimation
    # covariance
    Q = torch.eye(N, device=device) * q ** 2  # covariance of transition
    R = torch.eye(N, device=device) * r ** 2  # covariance of observation

    filter_model = filter_model_factory(robot, filter_model_name, device, Q, R)

    for i in range(T - 1):
        w = q * torch.randn(N, device=device)
        v = r * torch.randn(N, device=device)
        state[i + 1], obs[i] = robot(state[i] + w, input[i])  # model measurement
        if k is None:
            est[i + 1], P[i + 1] = filter_model(est[i], obs[i] + v, input[i], P[i])
        else:
            est[i + 1], P[i + 1] = filter_model(est[i], obs[i] + v, input[i], P[i], k=k)

    createPlot(filter_model_name, state, est, P)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filer Model Example')
    parser.add_argument("--filter_model_name", type=str, default='ukf', help="ekf or ukf")
    parser.add_argument("--device", type=str, default='cpu', help="cuda or cpu")
    parser.add_argument("--k", type=int, default=3, help="number")  # A parameter for
    # weighting the sigma points. When model_name=ekf, k=None.
    args = parser.parse_args()

    k = args.k
    device = args.device
    filter_model_name = args.filter_model_name
    T, N, M = 30, 3, 2  # steps, state dim, input dim
    q, r, p = 0.2, 0.2, 5  # covariance of transition noise, observation noise, and estimation
    bicycle = Bicycle()
    run_estimate(bicycle, k, T, N, M, q, r, p, filter_model_name, device)
