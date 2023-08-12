import cvxpy as cp
import numpy as np


def forward_single_np(Q, p, G, h, A, b):
    nz, neq, nineq = p.shape[0], A.shape[0] if A is not None else 0, G.shape[0]

    z_ = cp.Variable(nz)

    obj = cp.Minimize(0.5 * cp.quad_form(z_, Q) + p.T * z_)
    eqCon = A * z_ == b if neq > 0 else None
    if nineq > 0:
        slacks = cp.Variable(nineq)
        ineqCon = G * z_ + slacks == h
        slacksCon = slacks >= 0
    else:
        ineqCon = slacks = slacksCon = None
    cons = [x for x in [eqCon, ineqCon, slacksCon] if x is not None]
    prob = cp.Problem(obj, cons)
    prob.solve()  # solver=cp.SCS, max_iters=5000, verbose=False)
    # prob.solve(solver=cp.SCS, max_iters=10000, verbose=True)
    assert('optimal' in prob.status)
    zhat = np.array(z_.value).ravel()
    nu = np.array(eqCon.dual_value).ravel() if eqCon is not None else None
    if ineqCon is not None:
        lam = np.array(ineqCon.dual_value).ravel()
        slacks = np.array(slacks.value).ravel()
    else:
        lam = slacks = None

    return prob.value, zhat, nu, lam, slacks
