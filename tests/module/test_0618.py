import torch
import pypose as pp


n_state, n_ctrl = 4, 3
n_sc = n_state+n_ctrl
T = 3
n_batch = 2

expert_seed = 5
torch.manual_seed(expert_seed)

alpha = 0.2

expert = dict(
    Q = torch.tile(torch.eye(n_sc), (n_batch, T, 1, 1)),
    p = torch.tile(torch.randn(n_sc), (n_batch, T, 1)),
    A = torch.eye(n_state) \
        + 0.2 * torch.randn(n_state, n_state),
    B = torch.randn(n_state, n_ctrl),)


x_init = torch.randn(n_batch,n_state)
u_lower, u_upper = None, None
u_init = None

C = torch.eye(n_state)
D = torch.zeros(n_state, n_ctrl)
c1 = torch.zeros(n_state)
c2 = torch.zeros(n_state)
dt = 1

lti = pp.module.LTI(expert['A'], expert['B'], C, D, c1, c2)
mpc = pp.module.MPCO(lti, T, step=1)
x_true, u_true, cost_true = mpc.forward(expert['Q'], expert['p'], x_init, dt)

print(x_true)

u_lower = torch.tile(torch.randn(n_ctrl), (n_batch, T, 1))
u_upper = torch.tile(torch.randn(n_ctrl), (n_batch, T, 1))

lti = pp.module.LTI(expert['A'], expert['B'], C, D, c1, c2)
mpc = pp.module.MPC2(lti, T, step=1)
x_true, u_true, cost_true = mpc.forward(expert['Q'], expert['p'], x_init, dt, u_lower=u_lower, u_upper=u_upper)

print(x_true)
