import torch
import matplotlib.pyplot as plt
import pypose as pp
from pypose.function import bmv

class Quad(pp.module.NLS):
    """
    A simple 2D navigation model for testing MPPI on non-linear system
    """

    def __init__(self, dt, mass=7.4270, gravity=9.81):
        super().__init__()
        self._tau = dt
        self._m = mass
        self._g = gravity

    def state_transition(self, state, input, t=None):

        def dynamics(state, input):
            x, y, z, roll, pitch, yaw, vx, vy, vz = state.squeeze().moveaxis(-1, 0)
            thrust, mx, my, mz = input.squeeze().moveaxis(-1, 0)

            s_ph = torch.sin(roll)
            s_th = torch.sin(pitch)
            s_ps = torch.sin(yaw)
            c_ph = torch.cos(roll)
            c_th = torch.cos(pitch)
            c_ps = torch.cos(yaw)
            t_th = torch.tan(pitch)

            Rot_BI = torch.tensor([[c_th * c_ps,                      c_th * s_ps,                      -s_th],
                                   [s_ph * s_th * c_ps - c_ph * s_ps, s_ph * s_th * s_ps + c_ph * c_ps, s_ph * c_th],
                                   [c_ph * s_th * c_ps + s_ph * s_ps, c_ph * s_th * s_ps - s_ph * c_ps, c_ph * c_th]])


            nu = torch.tensor([[1, s_ph * t_th, c_ph * t_th],
                               [0., c_ph, -s_ph],
                               [0., s_ph/c_th, c_ph/c_th]])

            xDot = vx
            yDot = vy
            zDot = vz

            Omega = torch.stack((mx, my, mz), dim=-1)
            pDot = bmv(nu, Omega)
            r_Dot, p_Dot, y_Dot = pDot.squeeze().moveaxis(-1, 0)

            vDot = self._m * torch.tensor([0.0, 0.0, self._g]) - bmv(Rot_BI.mT, torch.tensor([0.0, 0.0, thrust]))
            vx_Dot, vy_Dot, vz_Dot = vDot.squeeze().moveaxis(-1, 0)

            _dstate = torch.stack((xDot, yDot, zDot, r_Dot, p_Dot, y_Dot, vx_Dot, vy_Dot, vz_Dot), dim=-1)

            return _dstate

        f1 = dynamics(state, input)
        f2 = dynamics(state + 0.5 * self._tau * f1, input)
        f3 = dynamics(state + 0.5 * self._tau * f2, input)
        f4 = dynamics(state + self._tau * f3, input)

        return (state.squeeze() + torch.mul(f1 + 2 * f2 + 2 * f3 + f4, self._tau/6.0)).unsqueeze(0)

    def observation(self, state, input, t=None):
        return state



def visualize(system, traj, controls):

    fig, axs = plt.subplots(3, 3, figsize=(18, 6))

    axs[0, 0].plot(traj[:, 0], label='x')
    axs[0, 1].plot(traj[:, 1], label='y')
    axs[0, 2].plot(traj[:, 2], label='z')
    axs[1, 0].plot(traj[:, 3], label='roll')
    axs[1, 1].plot(traj[:, 4], label='pitch')
    axs[1, 2].plot(traj[:, 5], label='yaw')
    axs[2, 0].plot(traj[:, 6], label='vx')
    axs[2, 1].plot(traj[:, 7], label='vy')
    axs[2, 2].plot(traj[:, 8], label='vz')

    axs[0, 0].set_title('x')
    axs[0, 1].set_title('y')
    axs[0, 2].set_title('z')
    axs[1, 0].set_title('roll')
    axs[1, 1].set_title('pitch')
    axs[1, 2].set_title('yaw')
    axs[2, 0].set_title('vx')
    axs[2, 1].set_title('vy')
    axs[2, 2].set_title('vz')

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

    plt.pause(system._tau)


if __name__ == '__main__':

    x_init = torch.tensor([[0., 0., -2., 0., 0., 0., 0., 0., 0.]], requires_grad=False)
    x_goal = torch.tensor([[1., -2., -3., 0., 0., 0., 0., 0., 0.]], requires_grad=False)

    dt = 0.02
    T = 5
    n_batch = 1
    n_state, n_ctrl = 9, 4

    u_init = torch.tile(torch.tensor([0.0, 0.0, 0.0, 0.0]), (n_batch, T, 1))

    Q = torch.tile(torch.eye(n_state + n_ctrl), (n_batch, T, 1, 1))
    Q[...,2,2], Q[...,3,3], Q[...,4,4], Q[...,5,5], Q[...,8,8] = 2, 0.5, 0.5, 0.5, 2
    Q[...,9,9], Q[...,10,10], Q[...,11,11], Q[...,12,12] = 0.5, 0.05, 0.05, 0.05
    #p = torch.tile(torch.ones(n_state + n_ctrl), (n_batch, T, 1))
    #print(Q)
    p = torch.tile(torch.zeros(n_state + n_ctrl), (n_batch, T, 1))
    dynamics=Quad(dt)
    stepper = pp.utils.ReduceToBason(steps=1, verbose=False)
    MPC = pp.module.MPC(dynamics, Q, p, T, stepper=stepper)

    N = 1

    xt = x_init

    X = [xt.squeeze()]
    U = [torch.tensor([7.4270*9.81, 0.0, 0.0, 0.0])]
    costs = []

    """ for i in range(N):
        x_init_mpc = xt - x_goal
        x_mpc, u_mpc, cost = MPC(dt, x_init_mpc, u_init=u_init)
        ut_mpc = u_mpc[...,0,:]
        print(ut_mpc)
        xt = dynamics.forward(xt, ut_mpc)[0]
        X.append(xt.squeeze())
        U.append(ut_mpc.squeeze()) """

    #visualize(Quad(dt, J), torch.stack(X), torch.stack(U))
    #plt.show(block=True)


    for i in range(N):
        x_init_mpc = xt - x_goal
        x_mpc, u_mpc, cost = MPC(dt, x_init_mpc, u_init=u_init)
        u_new = u_init + u_mpc
        #print("u",u_mpc)
        ut_new = u_new[...,0,:]
        xt = dynamics.forward(xt, ut_new)[0]
        u_next = u_new[...,1:,:]
        u_last = u_new[...,-1,:]
        u_init = torch.cat((u_next, u_last.unsqueeze(0)), dim=1)
        #print(u_init)
        X.append(xt.squeeze())
        U.append(ut_new.squeeze())

    #visualize(dynamics, torch.stack(X), torch.stack(U))
    #plt.show(block=True)
