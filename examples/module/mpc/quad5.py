import torch
import matplotlib.pyplot as plt
import pypose as pp
from pypose.function import bmv

class Quad(pp.module.NLS):
    """
    A simple 2D navigation model for testing MPPI on non-linear system
    """

    def __init__(self, dt, J, mass=1.0, gravity=9.81):
        super().__init__()
        self._tau = dt
        self._J = J
        self._m = mass
        self._g = gravity

    def state_transition(self, state, input, t=None):

        def dynamics(state, input):
            roll, pitch, yaw, wx, wy, wz = state.squeeze().moveaxis(-1, 0)
            thrust, mx, my, mz = input.squeeze().moveaxis(-1, 0)
            Ix, Iy, Iz = self._J[0,0], self._J[1,1], self._J[2,2]

            s_ph = torch.sin(roll)
            s_th = torch.sin(pitch)
            s_ps = torch.sin(yaw)
            c_ph = torch.cos(roll)
            c_th = torch.cos(pitch)
            c_ps = torch.cos(yaw)
            t_th = torch.tan(pitch)

            r_Dot = wx + wz * (c_ph * t_th) + wy * (s_ph * t_th)
            p_Dot = wy * c_ph - wz * s_ph
            y_Dot = wz * (c_ph/c_th) + wy * (s_ph/c_th)
            wx_Dot = (Iy - Iz)/Ix * wz * wy + mx/Ix
            wy_Dot = (Iz - Ix)/Iy * wx * wz + my/Iy
            wz_Dot = (Ix - Iy)/Iz * wx * wy + mz/Iz

            #_dstate = torch.stack((xDot, yDot, zDot, r_Dot, p_Dot, y_Dot, vx_Dot, vy_Dot, vz_Dot, wx_Dot, wy_Dot, wz_Dot), dim=-1)
            _dstate = torch.stack((r_Dot, p_Dot, y_Dot, wx_Dot, wy_Dot, wz_Dot), dim=-1)

            return _dstate

        f1 = dynamics(state, input)
        f2 = dynamics(state + 0.5 * self._tau * f1, input)
        f3 = dynamics(state + 0.5 * self._tau * f2, input)
        f4 = dynamics(state + self._tau * f3, input)

        return (state.squeeze() + torch.mul(f1 + 2 * f2 + 2 * f3 + f4, self._tau/6.0)).unsqueeze(0)

    def observation(self, state, input, t=None):
        return state


def visualize(system, traj, controls):

    fig, axs = plt.subplots(2, 3, figsize=(18, 6))

    axs[0, 0].plot(traj[:, 0], label='roll')
    axs[0, 1].plot(traj[:, 1], label='pitch')
    axs[0, 2].plot(traj[:, 2], label='yaw')
    axs[1, 0].plot(traj[:, 3], label='wx')
    axs[1, 1].plot(traj[:, 4], label='wy')
    axs[1, 2].plot(traj[:, 5], label='wz')

    axs[0, 0].set_title('roll')
    axs[0, 1].set_title('pitch')
    axs[0, 2].set_title('yaw')
    axs[1, 0].set_title('wx')
    axs[1, 1].set_title('wy')
    axs[1, 2].set_title('wz')

    fig.suptitle('states', fontsize=16)

    fig, axs = plt.subplots(2, 2, figsize=(18, 6))

    axs[0, 0].plot(controls[:, 0], label='thrust')
    axs[0, 1].plot(controls[:, 1], label='mx')
    axs[1, 0].plot(controls[:, 2], label='my')
    axs[1, 1].plot(controls[:, 3], label='mz')

    axs[0, 0].set_title('thrust')
    axs[0, 1].set_title('mx')
    axs[1, 0].set_title('my')
    axs[1, 1].set_title('mz')

    fig.suptitle('inputs', fontsize=16)

    plt.show(block=False)


if __name__ == '__main__':

    dt = 0.001
    T = 30
    n_batch = 1
    n_state, n_ctrl = 6, 4

    x_init = torch.tensor([[0.087, 0.087, 0.087, 0., 0., 0.]], requires_grad=False)
    x_goal = torch.tensor([[0., 0., 0., 0., 0., 0.]], requires_grad=False)

    u_init = torch.tile(torch.tensor([9.8, 0.0, 0.0, 0.0]), (n_batch, T, 1))

    J = torch.tensor([[0.01466, 0., 0.],
                      [0., 0.01466, 0.],
                      [0., 0., 0.02848]])

    Q = torch.tile(torch.eye(n_state + n_ctrl), (n_batch, T, 1, 1))
    Q[...,0,0], Q[...,1,1], Q[...,2,2] = 10000000, 1000000, 10000000
    Q[...,6,6], Q[...,7,7], Q[...,8,8], Q[...,9,9] = 0.01, 0.05, 0.05, 0.05
    #Q[...,10,10] = 1000000
    p = torch.tile(torch.zeros(n_state + n_ctrl), (n_batch, T, 1))
    dynamics=Quad(dt, J)
    stepper = pp.utils.ReduceToBason(steps=10, verbose=False)
    MPC = pp.module.MPC(dynamics, Q, p, T, stepper=stepper)

    N = 20

    xt = x_init

    X = [xt.squeeze()]
    U = []
    costs = []

    for i in range(N):
        x_init_mpc = xt - x_goal
        x_mpc, u_mpc, cost = MPC(dt, x_init_mpc, u_init=u_init)
        ut_mpc = u_mpc[...,0,:]
        #print(ut_mpc)
        xt = dynamics.forward(xt, ut_mpc)[0]
        X.append(xt.squeeze())
        U.append(ut_mpc.squeeze())
        #print(xt)

    visualize(dynamics, torch.stack(X), torch.stack(U))
    plt.show(block=True)
