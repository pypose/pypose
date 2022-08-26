import sys
sys.path.append("..")
import torch as torch
import pypose as pp
import cvxpy as cp
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def LQR_cp(C, c, F, f, x_init, T, n_state, n_ctrl): 

    tau = cp.Variable((n_state+n_ctrl, T))

    objs = []
    x0 = tau[:n_state,0]
    u0 = tau[n_state:,0]
    cons = [x0 == x_init]
    for t in range(T):
        xt = tau[:n_state,t]
        ut = tau[n_state:,t]
        objs.append(0.5*cp.quad_form(tau[:,t], C[t]) + cp.sum(cp.multiply(c[t], tau[:,t])))
        if t+1 < T:
            xtp1 = tau[:n_state, t+1]
            cons.append(xtp1 == F[t]@tau[:,t]+f[t])
    prob = cp.Problem(cp.Minimize(sum(objs)), cons)
    prob.solve()
    assert 'optimal' in prob.status
    return torch.tensor(tau.value), torch.tensor([obj_t.value for obj_t in objs])


def test_LQR_linear():
    torch.manual_seed(0)

    n_batch = 2
    n_state, n_ctrl = 3, 4
    n_sc = n_state + n_ctrl
    T = 5
    C = torch.randn(T, n_batch, n_sc, n_sc)
    C = torch.matmul(C.mT, C)
    c = torch.randn(T, n_batch, n_sc)
    alpha = 0.2
    R = torch.tile(torch.eye(n_state) + alpha * torch.randn(n_state, n_state), (T, n_batch, 1, 1))
    S = torch.tile(torch.randn(n_state, n_ctrl), (T, n_batch, 1, 1))
    F = torch.cat((R, S), axis=3)
    f = torch.tile(torch.randn(n_state), (T, n_batch, 1))
    x_init = torch.randn(n_batch, n_state)

    tau_cp, objs_cp = LQR_cp(C[:,0], c[:,0], F[:,0], f[:,0], x_init[0], T, n_state, n_ctrl)
    tau_cp = tau_cp.T
    x_cp = tau_cp[:,:n_state]
    u_cp = tau_cp[:,n_state:]

    C, c, R, S, F, f, x_init = [torch.Tensor(x).double() if x is not None else None
        for x in [C, c, R, S, F, f, x_init]]

    u_lqr = None
    LQR_DP  = pp.module.DP_LQR(n_state, n_ctrl, T, None, None)
    Ks, ks = LQR_DP.DP_LQR_backward(C, c, F, f)
    x_lqr, u_lqr, objs_lqr, tau = LQR_DP.DP_LQR_forward(x_init, C, c, F, f, Ks, ks)
    tau_lqr = torch.cat((x_lqr, u_lqr), 2)
    assert torch.allclose(tau_cp, tau_lqr[:,0]) 

    lambda_dual = LQR_DP.DP_LQR_costates(tau, C, c, F)
    
    lambda_dual_test = []
    Cs = []
    for t in range(T-1, -1, -1):
        t_rev_new = T-1-t
        tau_t = tau[t_rev_new]
        Ct = C[t]
        ct = c[t]
        Ft = F[t]
        Ct_x = Ct[:, :n_state, :]
        ct_x = ct[:, :n_state]

        if t == T-1:
            lambda_final_test = lambda_tp1_test = Ct_x.bmm(tau_t.unsqueeze(2)).squeeze(2) + ct_x
        else: 
            Ft_T = Ft.transpose(1,2)
            Ft_x_T = Ft_T[:, :n_state, :]
            lambda_t_test = Ft_x_T .bmm(lambda_tp1_test.unsqueeze(2)).squeeze(2) + Ct_x.bmm(tau_t.unsqueeze(2)).squeeze(2) + ct_x
            lambda_tp1_test = lambda_t_test
            lambda_dual_test.append(lambda_t_test)
            
    lambda_dual_test.reverse()
    lambda_dual_test.append(lambda_final_test)
    lambda_dual_test= torch.stack(lambda_dual_test)

    assert torch.allclose(lambda_dual, lambda_dual_test)

    print("Done")


if __name__ == '__main__':
    test_LQR_linear()
