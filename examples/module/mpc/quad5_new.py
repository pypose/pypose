import torch
import matplotlib.pyplot as plt
import pypose as pp
from pypose.function import bmv
import numpy as np

class Quad(pp.module.NLS):

    def __init__(self, dt,
                 J = torch.tensor([[0.1673, 0., 0.],
                                   [0., 0.1673, 0.],
                                   [0., 0., 0.2879]]), #inertia matrix
                  mass=7.4270, gravity=9.81):
        super().__init__()
        self._tau = dt
        self._J = J
        self._m = mass
        self._g = gravity

    def dstate_transition(self, state, input):
        roll, pitch, yaw, p, q, r, u, v, w, x, y, z = state.squeeze().moveaxis(-1, 0)
        #roll, pitch, yaw, p, q, r, vx, vy, vz, x, y, z = state.squeeze().moveaxis(-1, 0)
        '''
        [roll;pitch;yaw]: Euler angles, [ph;th;ps]
        [p;q;r]: angular velocity in the body-fixed frame
        [u;v;w]: velocity in the body frame
        [vx;vy;vz]: velocity in the world frame
        [x;y;x]: position in the world frame
        Setting the Z-axis downward as the positive direction
        '''
        thrust, mx, my, mz = input.squeeze().moveaxis(-1, 0)
        '''
        thrust: total thrust of the quadrotor
        mx, my, mz: torque in the world frame
        '''
        Ix, Iy, Iz = self._J[0,0], self._J[1,1], self._J[2,2]

        s_ph = torch.sin(roll)
        s_th = torch.sin(pitch)
        s_ps = torch.sin(yaw)
        c_ph = torch.cos(roll)
        c_th = torch.cos(pitch)
        c_ps = torch.cos(yaw)
        t_th = torch.tan(pitch)

        ph_Dot = p + r * (c_ph * t_th) + q * (s_ph * t_th)
        th_Dot = q * c_ph - r * s_ph
        ps_Dot = r * (c_ph/c_th) + q * (s_ph/c_th)
        p_Dot = (Iy - Iz)/Ix * r * q + mx/Ix
        q_Dot = (Iz - Ix)/Iy * p * r + my/Iy
        r_Dot = (Ix - Iy)/Iz * p * q + mz/Iz
        u_Dot = r * v - q * w - self._g * s_th
        v_Dot = p * w - r * u + self._g * (s_ph * c_th)
        w_Dot = q * u - p * v + self._g * (c_th * c_ph) - thrust/self._m
        x_Dot = w * (s_ph * s_ps + c_ph * c_ps * s_th) - v * (c_ph * s_ps - c_ps * s_ph * s_th) + u * (c_ps * c_th)
        y_Dot = v * (c_ph * c_ps + s_ph * s_ps * s_th) - w * (s_ph * c_ps - c_ph * s_ps * s_th) + u * (s_ps * c_th)
        z_Dot = w * (c_ph * c_th) - u * s_th + v * (c_th * s_ph)
        """ vx_Dot = - (thrust * (c_ph * s_th * c_ps + s_ph * s_ps)) / self._m
        vy_Dot = (thrust * (s_ph * c_ps - c_ph * s_th * s_ps)) / self._m
        vz_Dot = self._g - (thrust * (c_ph * c_th)) / self._m
        x_Dot = vx
        y_Dot = vy
        z_Dot = vz """

        _dstate = torch.stack((ph_Dot, th_Dot, ps_Dot, p_Dot, q_Dot, r_Dot, u_Dot, v_Dot, w_Dot, x_Dot, y_Dot, z_Dot), dim=-1)
        return _dstate


    def state_transition(self, state, input, t=None):

        # _dstate = self.dstate_transition(state, input)
        f1 = self.dstate_transition(state, input)
        f2 = self.dstate_transition(state + 0.5 * self._tau * f1, input)
        f3 = self.dstate_transition(state + 0.5 * self._tau * f2, input)
        f4 = self.dstate_transition(state + self._tau * f3, input)

        _dstate = torch.mul(f1 + 2 * f2 + 2 * f3 + f4, 1/6.0)

        return (state.squeeze() + torch.mul(_dstate, self._tau)).unsqueeze(0)

    def observation(self, state, input, t=None):
        return state


def visualize(system, traj, controls):

    fig, axs = plt.subplots(4, 3, figsize=(18, 6))

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
    axs[3, 2].set_title('wz')

    fig, axs = plt.subplots(1, 3, figsize=(18,8))

    #axs[0].plot(np.radians(traj[:, 0]), label='roll')
    #axs[1].plot(np.radians(traj[:, 1]), label='pitch')
    #axs[2].plot(np.radians(traj[:, 2]), label='yaw')

    axs[0].plot(np.arange(traj.size(0)) * system._tau, np.radians(traj[:, 0]), linewidth=2.2)
    axs[1].plot(np.arange(traj.size(0)) * system._tau, np.radians(traj[:, 1]), linewidth=2.2)
    axs[2].plot(np.arange(traj.size(0)) * system._tau, np.radians(traj[:, 2]), linewidth=2.2)

    axs[0].set_title('roll', fontsize=22)
    axs[1].set_title('pitch', fontsize=22)
    axs[2].set_title('yaw', fontsize=22)

    axs[0].set_xlabel('Time(s)', fontsize=22)
    axs[1].set_xlabel('Time(s)', fontsize=22)
    axs[2].set_xlabel('Time(s)', fontsize=22)

    axs[0].set_ylabel('radian(rad)', fontsize=22)
    axs[1].set_ylabel('radian(rad)', fontsize=22)
    axs[2].set_ylabel('radian(rad)', fontsize=22)

    axs[0].grid(True)
    axs[1].grid(True)
    axs[2].grid(True)

    axs[0].grid(which='major', alpha=0.5, linestyle='-')
    axs[1].grid(which='major', alpha=0.5, linestyle='-')
    axs[2].grid(which='major', alpha=0.5, linestyle='-')

    axs[0].tick_params(axis='x', labelsize=18)
    axs[1].tick_params(axis='x', labelsize=18)
    axs[2].tick_params(axis='x', labelsize=18)

    axs[0].tick_params(axis='y', labelsize=18)
    axs[1].tick_params(axis='y', labelsize=18)
    axs[2].tick_params(axis='y', labelsize=18)

    plt.subplots_adjust(wspace=1.4)

    plt.tight_layout()

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

    """ fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = traj[:, 9]
    y = traj[:, 10]
    z = traj[:, 11]

    ax.plot(x, y, z, linewidth=2.2)

    ax.set_xlabel('x(m)', fontsize=16)
    ax.set_ylabel('y(m)', fontsize=16)
    ax.set_zlabel('z(m)', fontsize=16)
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.tick_params(axis='z', labelsize=13)

    ax.scatter(0., 4., -9., color='red', label='Goal Position', s=100)

    ax.legend( fontsize=16) """

    plt.show(block=False)


if __name__ == '__main__':

    dt = 0.02
    T = 30
    n_batch = 1
    n_state, n_ctrl = 12, 4

    x_init = torch.tensor([[0.087, 0.087, 0.087, 0., 0., 0., 0., 0., 0., 0., 0., 0.]], requires_grad=False)
    x_goal = torch.tensor([[0., 0., 0., 0., 0., 0.,
                            0., 0., 0., 0, 0., 0.]], requires_grad=False)

    u_init = torch.tile(torch.tensor([9.81*7.4270, 0.0, 0.0, 0.0]), (n_batch, T, 1))

    Q = torch.tile(torch.eye(n_state + n_ctrl), (n_batch, T, 1, 1))
    Q[...,0,0], Q[...,1,1], Q[...,2,2] = 10000000, 1000000, 10000000
    Q[...,12,12], Q[...,13,13], Q[...,14,14], Q[...,15,15] = 0.01, 0.05, 0.05, 0.05
    p = torch.tile(torch.zeros(n_state + n_ctrl), (n_batch, T, 1))
    dynamics=Quad(dt)
    stepper = pp.utils.ReduceToBason(steps=5, verbose=False)
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
        print(xt)
        u_new = u_mpc[...,1:,:]
        u_last = u_mpc[...,-1,:]
        u_init = torch.cat((u_new, u_last.unsqueeze(0)), dim=1) #u shift, not necessary
        X.append(xt.squeeze())
        U.append(ut_mpc.squeeze())

    #print(xt)

    visualize(dynamics, torch.stack(X), torch.stack(U))
    plt.show(block=True)
