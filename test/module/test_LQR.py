import sys
sys.path.append("..")
import torch as torch
import torch.nn as nn
import pypose as pp
import numpy as np
import numpy.random as npr
import numpy.testing as npt
from numpy.testing import dec
import cvxpy as cp
from collections import namedtuple

QuadCost = namedtuple('QuadCost', 'C c')
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
        objs.append(0.5*cp.quad_form(tau[:,t], C[t]) +
                    cp.sum(cp.multiply(c[t], tau[:,t])))
        if t+1 < T:
            xtp1 = tau[:n_state, t+1]
            cons.append(xtp1 == F[t]*tau[:,t]+f[t])
    prob = cp.Problem(cp.Minimize(sum(objs)), cons)
    prob.solve()
    assert 'optimal' in prob.status
    return np.array(tau.value), np.array([obj_t.value for obj_t in objs])


def test_LQR_linear():
    npr.seed(1)

    n_batch = 2
    n_state, n_ctrl = 3, 4
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
    
    print(x_init)

    tau_cp, objs_cp = LQR_cp(
        C[:,0], c[:,0], F[:,0], f[:,0], x_init[0], T, n_state, n_ctrl)
    tau_cp = tau_cp.T
    x_cp = tau_cp[:,:n_state]
    u_cp = tau_cp[:,n_state:]

    C, c, R, S, F, f, x_init = [
        torch.Tensor(x).double() if x is not None else None
        for x in [C, c, R, S, F, f, x_init]
    ]

    dynamics = pp.module.LTI(R[0,0], S[0,0], f[0,0])

    u_lqr = None
    x_lqr, u_lqr, objs_lqr = pp.module.DP_LQR(
        n_state, n_ctrl, T, u_lqr,
        lqr_iter=10,
        backprop=False,
        verbose=1,
        exit_unconverged=True,
    )(x_init, QuadCost(C, c), dynamics)
    tau_lqr = torch.cat((x_lqr, u_lqr), 2)
    tau_lqr = tau_lqr
    npt.assert_allclose(tau_cp, tau_lqr[:,0].numpy(), rtol=1e-3)

    print(x_lqr())
    print("done")


if __name__ == '__main__':
    test_LQR_linear()
