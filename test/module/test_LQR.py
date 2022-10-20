import sys
sys.path.append("..")
import torch as torch
import pypose as pp
import cvxpy as cp
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def LQR_cp(Q, p, F, f, x_init, T, n_state, n_ctrl): 

    tau = cp.Variable((n_state+n_ctrl, T))

    objs = []
    x0 = tau[:n_state,0]
    u0 = tau[n_state:,0]
    cons = [x0 == x_init]
    for t in range(T):
        xt = tau[:n_state,t]
        ut = tau[n_state:,t]
        objs.append(0.5*cp.quad_form(tau[:,t], Q[t]) + cp.sum(cp.multiply(p[t], tau[:,t])))
        if t+1 < T:
            xtp1 = tau[:n_state, t+1]
            cons.append(xtp1 == F[t]@tau[:,t]+f[t])
    prob = cp.Problem(cp.Minimize(sum(objs)), cons)
    prob.solve()
    assert 'optimal' in prob.status
    return torch.tensor(tau.value), torch.tensor([obj_t.value for obj_t in objs])


def test_LQR_linear():

    torch.manual_seed(0)
    n_batch, n_state, n_ctrl, T = 2, 4, 3, 5
    n_sc = n_state + n_ctrl
    Q = torch.randn(T, n_batch, n_sc, n_sc)
    Q = torch.matmul(Q.mT, Q)
    p = torch.randn(T, n_batch, n_sc)
    alpha = 0.2
    A = torch.tile(torch.eye(n_state) + alpha * torch.randn(n_state, n_state), (n_batch, 1, 1))
    B = torch.tile(torch.randn(n_state, n_ctrl), (n_batch, 1, 1))
    C = torch.tile(torch.eye(n_state), (n_batch, 1, 1))
    D = torch.tile(torch.zeros(n_state, n_ctrl), (n_batch, 1, 1))
    c1 = torch.tile(torch.randn(n_state), (n_batch, 1))
    c2 = torch.tile(torch.zeros(n_state), (n_batch, 1))
    x_init = torch.randn(n_batch, n_state)
    u_lqr = torch.tile(torch.zeros(n_ctrl), (n_batch, 1))
    F = torch.cat((A, B), axis=2)
    F = torch.tile(F, (T, 1, 1, 1))
    f = torch.tile(c1, (T, 1, 1))

    tau_cp, objs_cp = LQR_cp(Q[:,0], p[:,0], F[:,0], f[:,0], x_init[0], T, n_state, n_ctrl)
    tau_cp = tau_cp.T
    x_cp = tau_cp[:,:n_state]
    u_cp = tau_cp[:,n_state:]

    Q, p, A, B, C, D, c1, c2, x_init, u_lqr = [torch.Tensor(x).double() if x is not None else None
        for x in [Q, p, A, B, C, D, c1, c2, x_init, u_lqr]]

    lti = pp.module.LTI(A, B, C, D, c1, c2)
    
    LQR_DP  = pp.module.DP_LQR(n_state, n_ctrl, T, lti)
    x_lqr, u_lqr, objs_lqr, tau = LQR_DP(x_init, Q, p)
    tau_lqr = torch.cat((x_lqr, u_lqr), 2)
    assert torch.allclose(tau_cp, tau_lqr[:,0]) 

    print("Done")


if __name__ == '__main__':
    test_LQR_linear()

