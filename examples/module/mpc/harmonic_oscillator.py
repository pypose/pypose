import torch
import matplotlib.pyplot as plt
import pypose as pp


if __name__ == '__main__':

    x_init = torch.tensor([[0., 0.]], requires_grad=False)
    x_goal = torch.tensor([[0.5, 0.]], requires_grad=False)
    dt = 0.01
    T = 5
    m = 1
    k = 100
    n_batch = 1
    n_state, n_ctrl = 2, 1

    Q = torch.tile(torch.eye(n_state + n_ctrl), (n_batch, T, 1, 1))
    p = torch.tile(torch.zeros(n_state + n_ctrl), (n_batch, T, 1))
    stepper = pp.utils.ReduceToBason(steps=1, verbose=False)

    A = torch.tile(torch.eye(n_state, n_state), (n_batch, 1, 1))
    A[...,0,1], A[...,1,0] = dt, -k*dt/m
    B = torch.zeros(n_batch, n_state, n_ctrl)
    B[...,1,0] = dt/m
    C = torch.tile(torch.eye(n_state), (n_batch, 1, 1))
    D = torch.zeros(n_batch, n_state, n_ctrl)
    c1 = torch.zeros(n_batch, n_state)
    c2 = torch.zeros(n_batch, n_state)

    lti = pp.module.LTI(A, B, C, D, c1, c2)
    MPC = pp.module.MPC(lti, Q, p, T, stepper=stepper)

    N = 1

    xt = x_init

    u_init = torch.arange(T)
    u_init = torch.tile(torch.arange(T), (n_batch, 1, 1)).transpose(1, 2).float()

    for i in range(N):
        x_init_mpc = xt - x_goal
        x_mpc, u_mpc, cost = MPC(dt, x_init_mpc, u_init=u_init)
        ut_mpc = u_mpc[...,0,:]
        xt = lti.forward(xt, ut_mpc)[0]
        u_new = u_mpc[...,1:,:]
        u_last = u_mpc[...,-1,:]
        u_init = torch.cat((u_new, u_last.unsqueeze(0)), dim=1)
