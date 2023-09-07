import torch
import matplotlib.pyplot as plt
import pypose as pp
import numpy as np

class Simple2DNav(pp.module.NLS):

    def __init__(self,dt,length=1.0):
        super().__init__()
        self._tau = dt
        self._length=length

    def state_transition(self, state, input, t=None):

        def dynamics(state, input):
            x, y, theta = state.squeeze().moveaxis(-1, 0)
            v, omega = input.squeeze().moveaxis(-1, 0)

            xDot = v * torch.cos(theta)
            yDot = v * torch.sin(theta)
            thetaDot = omega

            _dstate = torch.stack((xDot, yDot, thetaDot), dim=-1)

            return _dstate

        f1 = dynamics(state, input)
        f2 = dynamics(state + 0.5 * self._tau * f1, input)
        f3 = dynamics(state + 0.5 * self._tau * f2, input)
        f4 = dynamics(state + self._tau * f3, input)

        return (state.squeeze() + torch.mul(f1 + 2 * f2 + 2 * f3 + f4, self._tau/6.0)).unsqueeze(0)

    def observation(self, state, input, t=None):

        return state

def visualize(system, traj):

    """ fig, axs = plt.subplots(1, 2, figsize=(18, 6))
    plt.show(block=False)

    x = traj[:, 0]
    y = traj[:, 1]
    th = traj[:, 2]

    for i in range(traj.shape[0]):
        for ax in axs:
            ax.cla()

        axs[0].set_xlim(min(x)-1., max(x)+1.)
        axs[0].set_ylim(min(y)-1., max(y)+1.)

        axs[0].set_xlabel('x(m)')
        axs[0].set_ylabel('y(m)')

        axs[0].plot(x[:i], y[:i], alpha=0.3, c='b')
        axs[0].scatter(1.5, 1.5, c='r', label='Goal Position')


        for j, (label, color) in enumerate(zip(['x', 'y', 'th'], ['r', 'g', 'b'])):
            axs[1].plot(traj[:i, j], label=label, c=color)
            axs[1].scatter(i, traj[i, j], c=color, marker='.')

        for ax in axs:
            ax.legend()

        plt.pause(system._tau) """

    fig, ax = plt.subplots(figsize=(9, 9))

    x = traj[:, 0]
    y = traj[:, 1]

    ax.plot(x, y, alpha=0.3, c='b', linewidth=3.2)

    ax.set_xlim(min(x)-1., max(x)+1.)
    ax.set_ylim(min(y)-1., max(y)+1.)

    ax.set_xlabel('x(m)', fontsize=26)
    ax.set_ylabel('y(m)', fontsize=26)

    ax.scatter(1.5, 1.5, color='red', label='Goal Position', s=200)
    ax.legend(fontsize=16)

    for label in ax.legend().get_texts():
        label.set_fontsize(26)


    ax.tick_params(axis='both', which='major', labelsize=26)

    ax.set_aspect('equal')
    ax.grid(True)
    ax.grid(which='major', alpha=0.5, linestyle='-')

    x_ticks_interval = 0.5
    y_ticks_interval = 0.5
    ax.set_xticks(np.arange(min(x)-1., max(x) + 1.1, x_ticks_interval))
    ax.set_yticks(np.arange(min(y)-1., max(y) + 1., y_ticks_interval))

    plt.savefig('/Users/anaishe/Desktop/dubin car traj grid.png', bbox_inches='tight')
    plt.show(block=False)



if __name__ == '__main__':

    x_init = torch.tensor([[0., 0., 0.]], requires_grad=False)
    x_goal = torch.tensor([[1.5, 1.5, 0.]], requires_grad=False)
    dt = 0.1
    T = 25
    n_batch = 1
    n_state, n_ctrl = 3, 2

    Q = torch.tile(torch.eye(n_state + n_ctrl), (n_batch, T, 1, 1))
    Q[...,0,0], Q[...,1,1], Q[...,2,2], Q[...,3,3], Q[...,4,4] = 1, 6.55, 0.1, 0.5, 0.05
    #p = torch.tile(torch.ones(n_state + n_ctrl), (n_batch, T, 1))
    p = torch.tile(torch.zeros(n_state + n_ctrl), (n_batch, T, 1))
    dynamics=Simple2DNav(dt)
    stepper = pp.utils.ReduceToBason(steps=8, verbose=False)
    MPC = pp.module.MPC(dynamics, Q, p, T, stepper=stepper)

    N = 100

    xt = x_init

    """ X = [xt.squeeze()]
    U = [torch.tensor([0.,0.])]
    costs = []

    for i in range(N):
        x_init_mpc = xt - x_goal
        x_mpc, u_mpc, cost = MPC(dt, x_init_mpc)
        ut_mpc = u_mpc[...,0,:]
        xt = dynamics.forward(xt, ut_mpc)[0]
        X.append(xt.squeeze())
        U.append(ut_mpc.squeeze()) """


    u_init = torch.tile(torch.zeros(n_ctrl), (n_batch, T, 1))

    X = [xt.squeeze()]
    U = [torch.tensor([0.,0.])]
    costs = []

    for i in range(N):
        x_init_mpc = xt - x_goal
        x_mpc, u_mpc, cost = MPC(dt, x_init_mpc, u_init=u_init)
        ut_mpc = u_mpc[...,0,:]
        xt = dynamics.forward(xt, ut_mpc)[0]
        u_new = u_mpc[...,1:,:]
        u_last = u_mpc[...,-1,:]
        u_init = torch.cat((u_new, u_last.unsqueeze(0)), dim=1)
        X.append(xt.squeeze())
        U.append(ut_mpc.squeeze())

    print(xt)

    visualize(Simple2DNav(dt), torch.stack(X))
    plt.show(block=True)