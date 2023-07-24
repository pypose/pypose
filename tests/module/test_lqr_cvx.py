import torch
import numpy as np
import cvxpy as cp
import pypose as pp
import numpy.random as npr
from torch.autograd import Function, Variable, grad

class TestLQR:

    def test_lqr_linear_unbounded(self):

        npr.seed(1)

        n_batch = 3
        n_state, n_ctrl = 8, 4
        n_sc = n_state + n_ctrl
        T = 5
        C = npr.randn(T, n_batch, n_sc, n_sc)
        C = np.matmul(C.transpose(0, 1, 3, 2), C)
        c = npr.randn(T, n_batch, n_sc)
        alpha = 0.2
        R = np.tile(np.eye(n_state)+alpha*np.random.randn(n_state, n_state),
                    (T, n_batch, 1, 1))
        S = np.tile(np.random.randn(n_state, n_ctrl), (T, n_batch, 1, 1))
        F = np.concatenate((R, S), axis=3)
        f = np.tile(npr.randn(n_state), (T, n_batch, 1))
        x_init = npr.randn(n_batch, n_state)

        def lqr_cp(C, c, F, f, x_init, T, n_state, n_ctrl, u_lower, u_upper):
            """Solve min_{tau={x,u}} sum_t 0.5 tau_t^T C_t tau_t + c_t^T tau_t
                                s.t. x_{t+1} = A_t x_t + B_t u_t + f_t
                                    x_0 = x_init
                                    u_lower <= u <= u_upper
            """
            tau = cp.Variable((n_state+n_ctrl, T))
            assert (u_lower is None) == (u_upper is None)

            objs = []
            x0 = tau[:n_state,0]
            u0 = tau[n_state:,0]
            cons = [x0 == x_init]
            for t in range(T):
                xt = tau[:n_state,t]
                ut = tau[n_state:,t]
                objs.append(0.5*cp.quad_form(tau[:,t], C[t]) +
                            cp.sum(cp.multiply(c[t], tau[:,t])))
                if u_lower is not None:
                    cons += [u_lower[t] <= ut, ut <= u_upper[t]]
                if t+1 < T:
                    xtp1 = tau[:n_state, t+1]
                    cons.append(xtp1 == F[t]@tau[:,t]+f[t])
            prob = cp.Problem(cp.Minimize(sum(objs)), cons)
            # prob.solve(solver=cp.SCS, verbose=True)
            prob.solve()
            assert 'optimal' in prob.status
            return np.array(tau.value), np.array([obj_t.value for obj_t in objs])

        tau_cp, objs_cp = lqr_cp(
        C[:,0], c[:,0], F[:,0], f[:,0], x_init[0], T, n_state, n_ctrl,
        None, None
        )
        tau_cp = tau_cp.T #transpose
        x_cp = tau_cp[:,:n_state]
        u_cp = tau_cp[:,n_state:]

        C, c, R, S, F, f, x_init = [
        torch.tensor(x).to(torch.float32) if x is not None else None
        for x in [C, c, R, S, F, f, x_init]
        ]


        Q = C.permute(1, 0, 2, 3)
        p = c.permute(1, 0, 2)
        A = R.permute(1, 0, 2, 3)[:,0,:]
        B = S.permute(1, 0, 2, 3)[:,0,:]
        c1 = f.permute(1, 0, 2)[:,0,:]
        c2 = torch.zeros(n_batch, n_state)
        C = torch.tile(torch.eye(n_state), (n_batch, 1, 1))
        D = torch.zeros(n_batch, n_state, n_ctrl)

        lti = pp.module.LTI(A, B, C, D, c1, c2)
        LQR = pp.module.LQR(lti, Q, p, T)
        x_lqr, u_lqr, cost_lqr = LQR(x_init)

        torch.testing.assert_close(torch.tensor(x_cp).to(torch.float32), x_lqr[0,:T],
                                   rtol=1e-5, atol=1e-3)
        torch.testing.assert_close(torch.tensor(u_cp).to(torch.float32), u_lqr[0,:T],
                                   rtol=1e-5, atol=1e-3)


if __name__ == '__main__':
    test = TestLQR()
    test.test_lqr_linear_unbounded()
