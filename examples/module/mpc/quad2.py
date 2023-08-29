import torch
import matplotlib.pyplot as plt
import pypose as pp
from pypose.function import bmv
import numpy as np

class Quad(pp.module.NLS):

    def __init__(self, dt, J, mass=1.0, gravity=9.81):
        super().__init__()
        self._tau = dt
        self._J = J
        self._m = mass
        self._g = gravity

    def state_transition(self, state, input, t=None):

        def dynamics(state, input):
            roll, pitch, yaw, wx, wy, wz, vx, vy, vz, x, y, z = state.squeeze().moveaxis(-1, 0)
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
            vx_Dot = wz * vy - wy * vz - self._g * s_th
            vy_Dot = wx * vz - wz * vx + self._g * (s_ph * c_th)
            vz_Dot = wy * vx - wx * vy + self._g * (c_th * c_ph) - thrust/self._m
            xDot = vz * (s_ph * s_ps + c_ph * c_ps * s_th) - vy * (c_ph * s_ps - c_ps * s_ph * s_th) + vx * (c_ps * c_th)
            yDot = vy * (c_ph * c_ps + s_ph * s_ps * s_th) - vz * (s_ph * c_ps - c_ph * s_ps * s_th) + vx * (s_ps * c_th)
            zDot = vz * (c_ph * c_th) - vx * s_th + vy * (c_th * s_ph)

            #_dstate = torch.stack((xDot, yDot, zDot, r_Dot, p_Dot, y_Dot, vx_Dot, vy_Dot, vz_Dot, wx_Dot, wy_Dot, wz_Dot), dim=-1)
            _dstate = torch.stack((r_Dot, p_Dot, y_Dot, wx_Dot, wy_Dot, wz_Dot, vx_Dot, vy_Dot, vz_Dot, xDot, yDot, zDot), dim=-1)

            return _dstate

        f1 = dynamics(state, input)
        f2 = dynamics(state + 0.5 * self._tau * f1, input)
        f3 = dynamics(state + 0.5 * self._tau * f2, input)
        f4 = dynamics(state + self._tau * f3, input)

        return (state.squeeze() + torch.mul(f1 + 2 * f2 + 2 * f3 + f4, self._tau/6.0)).unsqueeze(0)

    def observation(self, state, input, t=None):
        return state


def visualize(system, traj, controls):


    """ fig, axs = plt.subplots(4, 3, figsize=(18, 6))

    axs[0, 0].plot(traj[:, 9], label='x')
    axs[0, 1].plot(traj[:, 10], label='y')
    axs[0, 2].plot(traj[:, 11], label='z')
    axs[1, 0].plot(traj[:, 0], label='roll')
    axs[1, 1].plot(traj[:, 1], label='pitch')
    axs[1, 2].plot(traj[:, 2], label='yaw')
    axs[2, 0].plot(traj[:, 6], label='vx')
    axs[2, 1].plot(traj[:, 7], label='vy')
    axs[2, 2].plot(traj[:, 8], label='vz')
    axs[3, 0].plot(traj[:, 3], label='wx')
    axs[3, 1].plot(traj[:, 4], label='wy')
    axs[3, 2].plot(traj[:, 5], label='wz')

    axs[0, 0].set_title('x')
    axs[0, 1].set_title('y')
    axs[0, 2].set_title('z')
    axs[1, 0].set_title('roll')
    axs[1, 1].set_title('pitch')
    axs[1, 2].set_title('yaw')
    axs[2, 0].set_title('vx')
    axs[2, 1].set_title('vy')
    axs[2, 2].set_title('vz')
    axs[3, 0].set_title('wx')
    axs[3, 1].set_title('wy')
    axs[3, 2].set_title('wz') """

    fig, axs = plt.subplots(1, 3, figsize=(18,8))

    #axs[0].plot(np.radians(traj[:, 0]), label='roll')
    #axs[1].plot(np.radians(traj[:, 1]), label='pitch')
    #axs[2].plot(np.radians(traj[:, 2]), label='yaw')

    axs[0].plot(np.arange(traj.size(0)) * system._tau, np.radians(traj[:, 0]), linewidth=3.2)
    axs[1].plot(np.arange(traj.size(0)) * system._tau, np.radians(traj[:, 1]), linewidth=3.2)
    axs[2].plot(np.arange(traj.size(0)) * system._tau, np.radians(traj[:, 2]), linewidth=3.2)

    axs[0].set_title('roll', fontsize=26)
    axs[1].set_title('pitch', fontsize=26)
    axs[2].set_title('yaw', fontsize=26)

    axs[0].set_xlabel('Time(s)', fontsize=26)
    axs[1].set_xlabel('Time(s)', fontsize=26)
    axs[2].set_xlabel('Time(s)', fontsize=26)

    axs[0].set_ylabel('radian(rad)', fontsize=26)
    axs[1].set_ylabel('radian(rad)', fontsize=26)
    axs[2].set_ylabel('radian(rad)', fontsize=26)

    axs[0].grid(True)
    axs[1].grid(True)
    axs[2].grid(True)

    axs[0].grid(which='major', alpha=0.5, linestyle='-')
    axs[1].grid(which='major', alpha=0.5, linestyle='-')
    axs[2].grid(which='major', alpha=0.5, linestyle='-')

    axs[0].tick_params(axis='x', labelsize=26)
    axs[1].tick_params(axis='x', labelsize=26)
    axs[2].tick_params(axis='x', labelsize=26)

    axs[0].tick_params(axis='y', labelsize=26)
    axs[1].tick_params(axis='y', labelsize=26)
    axs[2].tick_params(axis='y', labelsize=26)

    plt.subplots_adjust(wspace=1.4)

    plt.tight_layout()

    """ fig.suptitle('states', fontsize=16)

    fig, axs = plt.subplots(2, 2, figsize=(18, 6))

    axs[0, 0].plot(controls[:, 0], label='thrust')
    axs[0, 1].plot(controls[:, 1], label='mx')
    axs[1, 0].plot(controls[:, 2], label='my')
    axs[1, 1].plot(controls[:, 3], label='mz')

    axs[0, 0].set_title('thrust')
    axs[0, 1].set_title('mx')
    axs[1, 0].set_title('my')
    axs[1, 1].set_title('mz')

    fig.suptitle('inputs', fontsize=16) """

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')

    x = traj[:, 9]
    y = traj[:, 10]
    z = traj[:, 11]

    ax.plot(x, y, z, linewidth=3.2)

    ax.set_xlabel('x(m)', fontsize=26, labelpad=26)
    ax.set_ylabel('y(m)', fontsize=26, labelpad=26)
    ax.set_zlabel('z(m)', fontsize=26, labelpad=26)
    ax.tick_params(axis='x', labelsize=22)
    ax.tick_params(axis='y', labelsize=22)
    ax.tick_params(axis='z', labelsize=22)

    ax.scatter(0., 4., -9., color='red', label='Goal Position', s=200)

    ax.legend( fontsize=26)
    #plt.tight_layout()
    plt.savefig('/Users/anaishe/Desktop/quadrotor traj.png')

    plt.show(block=False)


if __name__ == '__main__':

    dt = 0.001
    T = 25
    n_batch = 1
    n_state, n_ctrl = 12, 4

    x_init = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 3.5, -8.]], requires_grad=False)
    x_goal = torch.tensor([[0., 0., 0., 0., 0., 0.,
                            0., 0., 0., 0, 4., -9.]], requires_grad=False)

    u_init = torch.tile(torch.tensor([9.81, 0.0, 0.0, 0.0]), (n_batch, T, 1))

    J = torch.tensor([[0.01466, 0., 0.],
                      [0., 0.01466, 0.],
                      [0., 0., 0.02848]])

    Q = torch.tile(torch.eye(n_state + n_ctrl), (n_batch, T, 1, 1))
    Q[...,0,0], Q[...,1,1], Q[...,2,2], Q[...,8,8] = 1000, 0.1, 0.1, 1000
    Q[...,12,12], Q[...,13,13], Q[...,14,14], Q[...,15,15] = 0.01, 0.05, 0.05, 0.05
    Q[...,10,10] = 70000000
    Q[...,11,11] = 20000000
    p = torch.tile(torch.zeros(n_state + n_ctrl), (n_batch, T, 1))
    dynamics=Quad(dt, J)
    stepper = pp.utils.ReduceToBason(steps=3, verbose=False)
    MPC = pp.module.MPC(dynamics, Q, p, T, stepper=stepper)

    N = 210

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

    print(xt)

    visualize(dynamics, torch.stack(X), torch.stack(U))
    plt.show(block=True)
