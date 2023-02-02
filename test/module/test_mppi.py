import math
import pypose as pp
import torch as torch
import matplotlib.pyplot as plt

from pypose.module.mppi import MPPI

# The class
class CartPole(pp.module.System):
    def __init__(self):
        super(CartPole, self).__init__()
        self._tau = 0.01
        self._length = 1.5
        self._cartmass = 20.0
        self._polemass = 0.1
        self._gravity = 9.81
        self._polemassLength = self._polemass * self._length
        self._totalMass = self._cartmass + self._polemass

    def state_transition(self, state, input, t=None):
        x, xDot, theta, thetaDot = state.moveaxis(-1, 0)
        force = input.squeeze().moveaxis(-1, 0)
        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)

        temp = (
            force + self._polemassLength * thetaDot**2 * sintheta
        ) / self._totalMass
        thetaAcc = (self._gravity * sintheta - costheta * temp) / (
            self._length * (4.0 / 3.0 - self._polemass * costheta**2 / self._totalMass)
        )
        xAcc = temp - self._polemassLength * thetaAcc * costheta / self._totalMass

        _dstate = torch.stack((xDot, xAcc, thetaDot, thetaAcc), dim=-1)

        return state + torch.mul(_dstate, self._tau)

    def observation(self, state, input, t=None):
        return state

def visualize(system, traj, controls):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    plt.show(block=False)

    l = system._length
    x = traj[:, 0]
    th = traj[:, 2]

    #plot cartpole
    for i in range(traj.shape[0]):
        for ax in axs:
            ax.cla()

        axs[0].set_xlim(x[i] - 3., x[i] + 3.)
        axs[0].set_ylim(-3., 3.)

        axs[0].plot(x, torch.zeros_like(x), alpha=0.3, c='r')
        axs[0].plot(x + l*torch.sin(th), l*torch.cos(th), alpha=0.3, c='k')

        axs[0].scatter(x[i], 0., c='r', label='com')
        axs[0].scatter(x[i] + l*torch.sin(th[i]), l*torch.cos(th[i]), c='k', label='endpt')
        
        #plot states
        for j, (label, color) in enumerate(zip(['x', 'xd', 'th', 'thd'], ['r', 'g', 'b', 'y'])):
            axs[1].plot(traj[:, j], label=label, c=color)
            axs[1].scatter(i, traj[i, j], marker='.', c=color)

        #plot control
        for j, label in enumerate(['fx']):
            axs[2].plot(controls[:, j], label=label, c='r')
            axs[2].scatter(i, controls[i, j], c='r', marker='.')

        for ax in axs:
            ax.legend()

        plt.pause(system._tau)

if __name__ == '__main__':
    import math
    cartpole = CartPole()
    cartpole._tau = 0.1

    ## test viz and simple traj ##
    dt = 0.1
    N = 200
    time = torch.arange(N).unsqueeze(-1) * dt
    U = torch.sin(time) * 0.
    x0 = torch.tensor([0., 0.5, 0.2, 0.2])
    X = [x0]
    for u in U:
        xc = X[-1].clone()
        xn = cartpole.forward(xc, u)[0]
        X.append(xn)

    X = torch.stack(X, dim=0)

#    visualize(cartpole, X, U)

    cost_fn = lambda x, u: (x[..., 1]) * 0.0 + (x[..., 1].pow(2)) * 100. + (x[..., 3].pow(2)) * 100.0

    mppi = MPPI(
        dynamics=cartpole,
        running_cost=cost_fn,
        nx=4,
        noise_sigma=torch.eye(1) * 1.0,
        num_samples=5000,
        horizon=100,
        lambda_=0.01,
        u_scale=1.0
    )

    X = [x0]
    U = []
    for i in range(N):
        xc = X[-1]
        u = mppi.command(xc)
        xn = cartpole.forward(xc, u)[0]
        X.append(xn)
        U.append(u)

    X = torch.stack(X, dim=0)
    U = torch.stack(U, dim=0)

    print('final cost: {:.2f}'.format(cost_fn(X, U).sum()))
    visualize(cartpole, X[1:], U)
