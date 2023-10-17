import torch
import pypose as pp
import matplotlib.pyplot as plt

class Simple2DNav(pp.module.System):
    """
    A simple 2D navigation model for testing MPPI on non-linear system
    """

    def __init__(self,dt,length=1.0):
        super().__init__()
        self._tau = dt
        self._length=length

    def state_transition(self, state, input, t=None):
        """
        The simple 2D nav has state: (x, y, theta) and input: (v, omega)
        """
        x, y, theta = state.moveaxis(-1, 0)
        v, omega = input.squeeze().moveaxis(-1, 0)
        xDot = v * torch.cos(theta)
        yDot = v * torch.sin(theta)
        thetaDot = omega
        _dstate = torch.stack((xDot, yDot, thetaDot), dim=-1)
        return (state.squeeze() + torch.mul(_dstate, self._tau)).unsqueeze(0)

    def observation(self, state, input, t=None):
        """
        Returns:
            [N x 3] Tensor of state (as the system is fully observable)
        """
        return state



def visualize(system, traj, controls, costs):
    """
    pyplot visualization of Simple2DNav for debugging
    Args:
        system: The Simple2DNav system
        traj: [T x 3] Tensor of states to plot
        controls: [T x 2] Tensor of controls to plot
        costs: [T] List of costs to plot
    Returns:
        None
    """

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    plt.show(block=False)

    x = traj[:, 0]
    y = traj[:, 1]
    th = traj[:, 2]

    #plot position
    for i in range(traj.shape[0]):
        for ax in axs:
            ax.cla()

        axs[0].set_xlim(min(x)-1., max(x)+1.)
        axs[0].set_ylim(min(y)-1., max(y)+1.)

        axs[0].plot(x[:i], y[:i], alpha=0.3, c='b')
        axs[0].scatter(x[i], y[i], c='r', label='Current Position')

        #plot states
        for j, (label, color) in enumerate(zip(['x', 'y', 'th'], ['r', 'g', 'b'])):
            axs[1].plot(traj[:i, j], label=label, c=color)
            axs[1].scatter(i, traj[i, j], c=color, marker='.')

        # Plot cost
        axs[2].plot(costs[:i], label='Cost', c='r')

        for ax in axs:
            ax.legend()

        plt.pause(system._tau)


if __name__ == '__main__':
    # Define initial state
    torch.manual_seed(0)
    x0 = torch.tensor([0., 0., 0.], requires_grad=False)
    dt=0.1

    cost_fn = lambda x, u, t: (x[..., 0] - 10)**2 + (x[..., 1] - 10)**2 + (u[..., 0])**2


    mppi = pp.MPPI(
        dynamics=Simple2DNav(dt),
        running_cost=cost_fn,
        nx=3,
        noise_sigma=torch.eye(2) * 1,
        num_samples=100,
        horizon=5,
        lambda_=0.01
        )

    N = 40
    X = [x0]
    U = []
    costs=[]
    i = 0
    xn=x0

    while abs((xn[0] - 10))>0.1 and  abs((xn[1] - 10)) > 0.1:
        xc = X[-1]
        u, xn= mppi.forward(xc)
        xn=xn[-1]
        costs.append(cost_fn(xc,u[-1],1))
        X.append(xn)
        U.append(u)
        i += 1
        if i > 100: break

    visualize(Simple2DNav(dt), torch.stack(X), torch.stack(U), costs)
    plt.show()
