import torch
import numpy as np

from qpth.util import get_sizes

# TODO: Add more comments describing the math here.
# https://stanford.edu/~boyd/papers/pdf/code_gen_impl.pdf


def forward(inputs_i, Q, G, A, b, h, U_Q, U_S, R, verbose=False):
    """
    b = A z_0
    h = G z_0 + s_0
    U_Q, U_S, R = pre_factor_kkt(Q, G, A, nineq, neq)
    """
    nineq, nz, neq, _ = get_sizes(G, A)

    # find initial values
    d = torch.ones(nineq).type_as(Q)
    nb = -b if b is not None else None
    factor_kkt(U_S, R, d)
    x, s, z, y = solve_kkt(
        U_Q, d, G, A, U_S,
        inputs_i, torch.zeros(nineq).type_as(Q), -h, nb)
    # x1, s1, z1, y1 = factor_solve_kkt(Q, torch.eye(nineq).type_as(Q), G, A, inputs_i,
    # torch.zeros(nineq).type_as(Q), -h, nb)

    if torch.min(s) < 0:
        s -= torch.min(s) - 1
    if torch.min(z) < 0:
        z -= torch.min(z) - 1

    prev_resid = None
    for i in range(20):
        # affine scaling direction
        rx = (torch.mv(A.t(), y) if neq > 0 else 0.) + \
            torch.mv(G.t(), z) + torch.mv(Q, x) + inputs_i
        rs = z
        rz = torch.mv(G, x) + s - h
        ry = torch.mv(A, x) - b if neq > 0 else torch.Tensor([0.])
        mu = torch.dot(s, z) / nineq
        pri_resid = torch.norm(ry) + torch.norm(rz)
        dual_resid = torch.norm(rx)
        resid = pri_resid + dual_resid + nineq * mu
        d = z / s
        if verbose:
            print(("primal_res = {0:.5g}, dual_res = {1:.5g}, " +
                   "gap = {2:.5g}, kappa(d) = {3:.5g}").format(
                pri_resid, dual_resid, mu, min(d) / max(d)))
        # if (pri_resid < 5e-4 and dual_resid < 5e-4 and mu < 4e-4):
        improved = (prev_resid is None) or (resid < prev_resid + 1e-6)
        if not improved or resid < 1e-6:
            return x, y, z
        prev_resid = resid

        factor_kkt(U_S, R, d)
        dx_aff, ds_aff, dz_aff, dy_aff = solve_kkt(U_Q, d, G, A, U_S,
                                                   rx, rs, rz, ry)
        # D = torch.diag((z/s).cpu()).type_as(Q)
        # dx_aff1, ds_aff1, dz_aff1, dy_aff1 = factor_solve_kkt(Q, D, G, A, rx, rs, rz, ry)

        # compute centering directions
        alpha = min(min(get_step(z, dz_aff), get_step(s, ds_aff)), 1.0)
        sig = (torch.dot(s + alpha * ds_aff, z +
                         alpha * dz_aff) / (torch.dot(s, z)))**3
        dx_cor, ds_cor, dz_cor, dy_cor = solve_kkt(
            U_Q, d, G, A, U_S, torch.zeros(nz).type_as(Q),
            (-mu * sig * torch.ones(nineq).type_as(Q) + ds_aff * dz_aff) / s,
            torch.zeros(nineq).type_as(Q), torch.zeros(neq).type_as(Q))
        # dx_cor, ds_cor, dz_cor, dy_cor = factor_solve_kkt(Q, D, G, A,
        #     torch.zeros(nz).type_as(Q),
        #     (-mu*sig*torch.ones(nineq).type_as(Q) + ds_aff*dz_aff)/s,
        #     torch.zeros(nineq).type_as(Q), torch.zeros(neq).type_as(Q))

        dx = dx_aff + dx_cor
        ds = ds_aff + ds_cor
        dz = dz_aff + dz_cor
        dy = dy_aff + dy_cor if neq > 0 else None
        alpha = min(1.0, 0.999 * min(get_step(s, ds), get_step(z, dz)))
        dx_norm = torch.norm(dx)
        dz_norm = torch.norm(dz)
        if np.isnan(dx_norm) or dx_norm > 1e5 or dz_norm > 1e5:
            # Overflow, return early
            return x, y, z

        x += alpha * dx
        s += alpha * ds
        z += alpha * dz
        y = y + alpha * dy if neq > 0 else None

    return x, y, z


def get_step(v, dv):
    I = dv < 1e-12
    if torch.sum(I) > 0:  # TODO: Use something like torch.any(dv < 0)
        a = -v / dv
        return torch.min(a[I])
    else:
        return 1


def solve_kkt(U_Q, d, G, A, U_S, rx, rs, rz, ry, dbg=False):
    """ Solve KKT equations for the affine step"""
    nineq, nz, neq, _ = get_sizes(G, A)

    invQ_rx = torch.potrs(rx.view(-1, 1), U_Q).view(-1)
    if neq > 0:
        h = torch.cat([torch.mv(A, invQ_rx) - ry,
                       torch.mv(G, invQ_rx) + rs / d - rz], 0)
    else:
        h = torch.mv(G, invQ_rx) + rs / d - rz

    w = -torch.potrs(h.view(-1, 1), U_S).view(-1)

    g1 = -rx - torch.mv(G.t(), w[neq:])
    if neq > 0:
        g1 -= torch.mv(A.t(), w[:neq])
    g2 = -rs - w[neq:]

    dx = torch.potrs(g1.view(-1, 1), U_Q).view(-1)
    ds = g2 / d
    dz = w[neq:]
    dy = w[:neq] if neq > 0 else None

    # if np.all(np.array([x.norm() for x in [rx, rs, rz, ry]]) != 0):
    if dbg:
        import IPython
        import sys
        IPython.embed()
        sys.exit(-1)

    # if rs.norm() > 0: import IPython, sys; IPython.embed(); sys.exit(-1)
    return dx, ds, dz, dy


def pre_factor_kkt(Q, G, A):
    """ Perform all one-time factorizations and cache relevant matrix products"""
    nineq, nz, neq, _ = get_sizes(G, A)

    # S = [ A Q^{-1} A^T        A Q^{-1} G^T           ]
    #     [ G Q^{-1} A^T        G Q^{-1} G^T + D^{-1} ]

    U_Q = torch.potrf(Q)
    # partial cholesky of S matrix
    U_S = torch.zeros(neq + nineq, neq + nineq).type_as(Q)

    G_invQ_GT = torch.mm(G, torch.potrs(G.t(), U_Q))
    R = G_invQ_GT
    if neq > 0:
        invQ_AT = torch.potrs(A.t(), U_Q)
        A_invQ_AT = torch.mm(A, invQ_AT)
        G_invQ_AT = torch.mm(G, invQ_AT)

        # TODO: torch.potrf sometimes says the matrix is not PSD but
        # numpy does? I filed an issue at
        # https://github.com/pytorch/pytorch/issues/199
        try:
            U11 = torch.potrf(A_invQ_AT)
        except:
            U11 = torch.Tensor(np.linalg.cholesky(
                A_invQ_AT.cpu().numpy())).type_as(A_invQ_AT)

        # TODO: torch.trtrs is currently not implemented on the GPU
        # and we are using gesv as a workaround.
        U12 = torch.gesv(G_invQ_AT.t(), U11.t())[0]
        U_S[:neq, :neq] = U11
        U_S[:neq, neq:] = U12
        R -= torch.mm(U12.t(), U12)

    return U_Q, U_S, R


def factor_kkt(U_S, R, d):
    """ Factor the U22 block that we can only do after we know D. """
    nineq = R.size(0)
    U_S[-nineq:, -nineq:] = torch.potrf(R + torch.diag(1 / d.cpu()).type_as(d))


def factor_solve_kkt(Q, D, G, A, rx, rs, rz, ry):
    nineq, nz, neq, _ = get_sizes(G, A)

    if neq > 0:
        H_ = torch.cat([torch.cat([Q, torch.zeros(nz, nineq).type_as(Q)], 1),
                        torch.cat([torch.zeros(nineq, nz).type_as(Q), D], 1)], 0)
        A_ = torch.cat([torch.cat([G, torch.eye(nineq).type_as(Q)], 1),
                        torch.cat([A, torch.zeros(neq, nineq).type_as(Q)], 1)], 0)
        g_ = torch.cat([rx, rs], 0)
        h_ = torch.cat([rz, ry], 0)
    else:
        H_ = torch.cat([torch.cat([Q, torch.zeros(nz, nineq).type_as(Q)], 1),
                        torch.cat([torch.zeros(nineq, nz).type_as(Q), D], 1)], 0)
        A_ = torch.cat([G, torch.eye(nineq).type_as(Q)], 1)
        g_ = torch.cat([rx, rs], 0)
        h_ = rz

    U_H_ = torch.potrf(H_)

    invH_A_ = torch.potrs(A_.t(), U_H_)
    invH_g_ = torch.potrs(g_.view(-1, 1), U_H_).view(-1)

    S_ = torch.mm(A_, invH_A_)
    U_S_ = torch.potrf(S_)
    t_ = torch.mv(A_, invH_g_).view(-1, 1) - h_
    w_ = -torch.potrs(t_, U_S_).view(-1)
    v_ = torch.potrs(-g_.view(-1, 1) - torch.mv(A_.t(), w_), U_H_).view(-1)

    return v_[:nz], v_[nz:], w_[:nineq], w_[nineq:] if neq > 0 else None
