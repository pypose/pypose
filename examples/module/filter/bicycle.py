import torch, os
import numpy as np
import pypose as pp
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.legend_handler import HandlerLine2D


class Bicycle(pp.module.NLS):
    '''
    This is an implementation of the 2D Bicycle kinematic model,
    see: https://dingyan89.medium.com/simple-understanding-of-kinematic-bicycle-model
    -81cac6420357
    The robot is given a rotational and forward velocity, and traverses the 2D plane accordingly.
    This model is discrete-time non-linear system (NLS) and can be filtered with the
    -``pp.module.EKF`` and  ``pp.module.UKF``.
    '''

    def __init__(self):
        super().__init__()

    def state_transition(self, state, input, t=None):
        '''
        Don't add noise in this function, as it will be used for automatically
        linearizing the system by the parent class ``pp.module.NLS``.
        '''
        theta = state[..., 2] + input[1]
        x = state[..., 0] + input[..., 0] * theta.cos()
        y = state[..., 1] + input[..., 0] * theta.sin()
        return torch.stack([x, y, theta], dim=-1)

    def observation(self, state, input, t=None):
        '''
        Don't add noise in this function, as it will be used for automatically
        linearizing the system by the parent class ``pp.module.NLS``.
        '''
        return state


def bicycle_plot(model_name, state, est, cov, save=None, show=False):
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

    if save is not None:
        figure = os.path.join(save, model_name + '_bicycle.png')
        plt.savefig(figure)
        print("Saved to", figure)

    if show:
        plt.show()
