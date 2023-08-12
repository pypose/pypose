import torch
from enum import Enum
# from block import block

from qpth.util import get_sizes, bdiag


def lu_hack(x):
    data, pivots = x.lu(pivot=not x.is_cuda)
    if x.is_cuda:
        if x.ndimension() == 2:
            pivots = torch.arange(1, 1+x.size(0)).int().cuda()
        elif x.ndimension() == 3:
            pivots = torch.arange(
                1, 1+x.size(1),
            ).unsqueeze(0).repeat(x.size(0), 1).int().cuda()
        else:
            assert False
    return (data, pivots)


INACC_ERR = """
--------
qpth warning: Returning an inaccurate and potentially incorrect solution.

Some residual is large.
Your problem may be infeasible or difficult.

You can try using the CVXPY solver to see if your problem is feasible
and you can use the verbose option to check the convergence status of
our solver while increasing the number of iterations.

Advanced users:
You can also try to enable iterative refinement in the solver:
https://github.com/locuslab/qpth/issues/6
--------
"""


class KKTSolvers(Enum):
    LU_FULL = 1
    LU_PARTIAL = 2
    IR_UNOPT = 3


def forward(Q, p, G, h, A, b, Q_LU, S_LU, R, eps=1e-12, verbose=0, notImprovedLim=3,
            maxIter=20, solver=KKTSolvers.LU_PARTIAL):
    """
    Q_LU, S_LU, R = pre_factor_kkt(Q, G, A)
    """
    nineq, nz, neq, nBatch = get_sizes(G, A)

    # Find initial values
    if solver == KKTSolvers.LU_FULL:
        D = torch.eye(nineq).repeat(nBatch, 1, 1).type_as(Q)
        x, s, z, y = factor_solve_kkt(
            Q, D, G, A, p,
            torch.zeros(nBatch, nineq).type_as(Q),
            -h, -b if b is not None else None)
    elif solver == KKTSolvers.LU_PARTIAL:
        d = torch.ones(nBatch, nineq).type_as(Q)
        factor_kkt(S_LU, R, d)
        x, s, z, y = solve_kkt(
            Q_LU, d, G, A, S_LU,
            p, torch.zeros(nBatch, nineq).type_as(Q),
            -h, -b if neq > 0 else None)
    elif solver == KKTSolvers.IR_UNOPT:
        D = torch.eye(nineq).repeat(nBatch, 1, 1).type_as(Q)
        x, s, z, y = solve_kkt_ir(
            Q, D, G, A, p,
            torch.zeros(nBatch, nineq).type_as(Q),
            -h, -b if b is not None else None)
    else:
        assert False

    # Make all of the slack variables >= 1.
    M = torch.min(s, 1)[0]
    M = M.view(M.size(0), 1).repeat(1, nineq)
    I = M < 0
    s[I] -= M[I] - 1

    # Make all of the inequality dual variables >= 1.
    M = torch.min(z, 1)[0]
    M = M.view(M.size(0), 1).repeat(1, nineq)
    I = M < 0
    z[I] -= M[I] - 1

    best = {'resids': None, 'x': None, 'z': None, 's': None, 'y': None}
    nNotImproved = 0

    for i in range(maxIter):
        # affine scaling direction
        rx = (torch.bmm(y.unsqueeze(1), A).squeeze(1) if neq > 0 else 0.) + \
            torch.bmm(z.unsqueeze(1), G).squeeze(1) + \
            torch.bmm(x.unsqueeze(1), Q.transpose(1, 2)).squeeze(1) + \
            p
        rs = z
        rz = torch.bmm(x.unsqueeze(1), G.transpose(1, 2)).squeeze(1) + s - h
        ry = torch.bmm(x.unsqueeze(1), A.transpose(
            1, 2)).squeeze(1) - b if neq > 0 else 0.0
        mu = torch.abs((s * z).sum(1).squeeze() / nineq)
        z_resid = torch.norm(rz, 2, 1).squeeze()
        y_resid = torch.norm(ry, 2, 1).squeeze() if neq > 0 else 0
        pri_resid = y_resid + z_resid
        dual_resid = torch.norm(rx, 2, 1).squeeze()
        resids = pri_resid + dual_resid + nineq * mu

        d = z / s
        try:
            factor_kkt(S_LU, R, d)
        except:
            return best['x'], best['y'], best['z'], best['s']

        if verbose == 1:
            print('iter: {}, pri_resid: {:.5e}, dual_resid: {:.5e}, mu: {:.5e}'.format(
                i, pri_resid.mean(), dual_resid.mean(), mu.mean()))
        if best['resids'] is None:
            best['resids'] = resids
            best['x'] = x.clone()
            best['z'] = z.clone()
            best['s'] = s.clone()
            best['y'] = y.clone() if y is not None else None
            nNotImproved = 0
        else:
            I = resids < best['resids']
            if I.sum() > 0:
                nNotImproved = 0
            else:
                nNotImproved += 1
            I_nz = I.repeat(nz, 1).t()
            I_nineq = I.repeat(nineq, 1).t()
            best['resids'][I] = resids[I]
            best['x'][I_nz] = x[I_nz]
            best['z'][I_nineq] = z[I_nineq]
            best['s'][I_nineq] = s[I_nineq]
            if neq > 0:
                I_neq = I.repeat(neq, 1).t()
                best['y'][I_neq] = y[I_neq]
        if nNotImproved == notImprovedLim or best['resids'].max() < eps or mu.min() > 1e32:
            if best['resids'].max() > 1. and verbose >= 0:
                print(INACC_ERR)
            return best['x'], best['y'], best['z'], best['s']

        if solver == KKTSolvers.LU_FULL:
            D = bdiag(d)
            dx_aff, ds_aff, dz_aff, dy_aff = factor_solve_kkt(
                Q, D, G, A, rx, rs, rz, ry)
        elif solver == KKTSolvers.LU_PARTIAL:
            dx_aff, ds_aff, dz_aff, dy_aff = solve_kkt(
                Q_LU, d, G, A, S_LU, rx, rs, rz, ry)
        elif solver == KKTSolvers.IR_UNOPT:
            D = bdiag(d)
            dx_aff, ds_aff, dz_aff, dy_aff = solve_kkt_ir(
                Q, D, G, A, rx, rs, rz, ry)
        else:
            assert False

        # compute centering directions
        alpha = torch.min(torch.min(get_step(z, dz_aff),
                                    get_step(s, ds_aff)),
                          torch.ones(nBatch).type_as(Q))
        alpha_nineq = alpha.repeat(nineq, 1).t()
        t1 = s + alpha_nineq * ds_aff
        t2 = z + alpha_nineq * dz_aff
        t3 = torch.sum(t1 * t2, 1).squeeze()
        t4 = torch.sum(s * z, 1).squeeze()
        sig = (t3 / t4)**3

        rx = torch.zeros(nBatch, nz).type_as(Q)
        rs = ((-mu * sig).repeat(nineq, 1).t() + ds_aff * dz_aff) / s
        rz = torch.zeros(nBatch, nineq).type_as(Q)
        ry = torch.zeros(nBatch, neq).type_as(Q) if neq > 0 else torch.Tensor()

        if solver == KKTSolvers.LU_FULL:
            D = bdiag(d)
            dx_cor, ds_cor, dz_cor, dy_cor = factor_solve_kkt(
                Q, D, G, A, rx, rs, rz, ry)
        elif solver == KKTSolvers.LU_PARTIAL:
            dx_cor, ds_cor, dz_cor, dy_cor = solve_kkt(
                Q_LU, d, G, A, S_LU, rx, rs, rz, ry)
        elif solver == KKTSolvers.IR_UNOPT:
            D = bdiag(d)
            dx_cor, ds_cor, dz_cor, dy_cor = solve_kkt_ir(
                Q, D, G, A, rx, rs, rz, ry)
        else:
            assert False

        dx = dx_aff + dx_cor
        ds = ds_aff + ds_cor
        dz = dz_aff + dz_cor
        dy = dy_aff + dy_cor if neq > 0 else None
        alpha = torch.min(0.999 * torch.min(get_step(z, dz),
                                            get_step(s, ds)),
                          torch.ones(nBatch).type_as(Q))
        alpha_nineq = alpha.repeat(nineq, 1).t()
        alpha_neq = alpha.repeat(neq, 1).t() if neq > 0 else None
        alpha_nz = alpha.repeat(nz, 1).t()

        x += alpha_nz * dx
        s += alpha_nineq * ds
        z += alpha_nineq * dz
        y = y + alpha_neq * dy if neq > 0 else None

    if best['resids'].max() > 1. and verbose >= 0:
        print(INACC_ERR)
    return best['x'], best['y'], best['z'], best['s']


def get_step(v, dv):
    a = -v / dv
    a[dv > 0] = max(1.0, a.max())
    return a.min(1)[0].squeeze()


def unpack_kkt(v, nz, nineq, neq):
    i = 0
    x = v[:, i:i + nz]
    i += nz
    s = v[:, i:i + nineq]
    i += nineq
    z = v[:, i:i + nineq]
    i += nineq
    y = v[:, i:i + neq]
    return x, s, z, y


def kkt_resid_reg(Q_tilde, D_tilde, G, A, eps, dx, ds, dz, dy, rx, rs, rz, ry):
    dx, ds, dz, dy, rx, rs, rz, ry = [
        x.unsqueeze(2) if x is not None else None for x in
        [dx, ds, dz, dy, rx, rs, rz, ry]
    ]
    resx = Q_tilde.bmm(dx) + G.transpose(1, 2).bmm(dz) + rx
    if dy is not None:
        resx += A.transpose(1, 2).bmm(dy)
    ress = D_tilde.bmm(ds) + dz + rs
    resz = G.bmm(dx) + ds - eps * dz + rz
    resy = A.bmm(dx) - eps * dy + ry if dy is not None else None
    resx, ress, resz, resy = (
        v.squeeze(2) if v is not None else None for v in (resx, ress, resz, resy))
    return resx, ress, resz, resy


def solve_kkt_ir(Q, D, G, A, rx, rs, rz, ry, niter=1):
    """Inefficient iterative refinement."""
    nineq, nz, neq, nBatch = get_sizes(G, A)

    eps = 1e-7
    Q_tilde = Q + eps * torch.eye(nz).type_as(Q).repeat(nBatch, 1, 1)
    D_tilde = D + eps * torch.eye(nineq).type_as(Q).repeat(nBatch, 1, 1)

    dx, ds, dz, dy = factor_solve_kkt_reg(
        Q_tilde, D_tilde, G, A, rx, rs, rz, ry, eps)
    res = kkt_resid_reg(Q, D, G, A, eps,
                        dx, ds, dz, dy, rx, rs, rz, ry)
    resx, ress, resz, resy = res
    res = resx
    for k in range(niter):
        ddx, dds, ddz, ddy = factor_solve_kkt_reg(Q_tilde, D_tilde, G, A, -resx, -ress, -resz,
                                                  -resy if resy is not None else None,
                                                  eps)
        dx, ds, dz, dy = [v + dv if v is not None else None
                          for v, dv in zip((dx, ds, dz, dy), (ddx, dds, ddz, ddy))]
        res = kkt_resid_reg(Q, D, G, A, eps,
                            dx, ds, dz, dy, rx, rs, rz, ry)
        resx, ress, resz, resy = res
        # res = torch.cat(resx)
        res = resx

    return dx, ds, dz, dy


def factor_solve_kkt_reg(Q_tilde, D, G, A, rx, rs, rz, ry, eps):
    nineq, nz, neq, nBatch = get_sizes(G, A)

    H_ = torch.zeros(nBatch, nz + nineq, nz + nineq).type_as(Q_tilde)
    H_[:, :nz, :nz] = Q_tilde
    H_[:, -nineq:, -nineq:] = D
    if neq > 0:
        # H_ = torch.cat([torch.cat([Q, torch.zeros(nz,nineq).type_as(Q)], 1),
        # torch.cat([torch.zeros(nineq, nz).type_as(Q), D], 1)], 0)
        A_ = torch.cat([torch.cat([G, torch.eye(nineq).type_as(Q_tilde).repeat(nBatch, 1, 1)], 2),
                        torch.cat([A, torch.zeros(nBatch, neq, nineq).type_as(Q_tilde)], 2)], 1)
        g_ = torch.cat([rx, rs], 1)
        h_ = torch.cat([rz, ry], 1)
    else:
        A_ = torch.cat(
            [G, torch.eye(nineq).type_as(Q_tilde).repeat(nBatch, 1, 1)], 2)
        g_ = torch.cat([rx, rs], 1)
        h_ = rz

    H_LU = lu_hack(H_)

    invH_A_ = A_.transpose(1, 2).lu_solve(*H_LU)
    invH_g_ = g_.unsqueeze(2).lu_solve(*H_LU).squeeze(2)

    S_ = torch.bmm(A_, invH_A_)
    S_ -= eps * torch.eye(neq + nineq).type_as(Q_tilde).repeat(nBatch, 1, 1)
    S_LU = lu_hack(S_)
    t_ = torch.bmm(invH_g_.unsqueeze(1), A_.transpose(1, 2)).squeeze(1) - h_
    w_ = -t_.unsqueeze(2).lu_solve(*S_LU).squeeze(2)
    t_ = -g_ - w_.unsqueeze(1).bmm(A_).squeeze()
    v_ = t_.unsqueeze(2).lu_solve(*H_LU).squeeze(2)

    dx = v_[:, :nz]
    ds = v_[:, nz:]
    dz = w_[:, :nineq]
    dy = w_[:, nineq:] if neq > 0 else None

    return dx, ds, dz, dy


def factor_solve_kkt(Q, D, G, A, rx, rs, rz, ry):
    nineq, nz, neq, nBatch = get_sizes(G, A)

    H_ = torch.zeros(nBatch, nz + nineq, nz + nineq).type_as(Q)
    H_[:, :nz, :nz] = Q
    H_[:, -nineq:, -nineq:] = D
    if neq > 0:
        A_ = torch.cat([torch.cat([G, torch.eye(nineq).type_as(Q).repeat(nBatch, 1, 1)], 2),
                        torch.cat([A, torch.zeros(nBatch, neq, nineq).type_as(Q)], 2)], 1)
        g_ = torch.cat([rx, rs], 1)
        h_ = torch.cat([rz, ry], 1)
    else:
        A_ = torch.cat([G, torch.eye(nineq).type_as(Q)], 1)
        g_ = torch.cat([rx, rs], 1)
        h_ = rz

    H_LU = lu_hack(H_)

    invH_A_ = A_.transpose(1, 2).lu_solve(*H_LU)
    invH_g_ = g_.unsqueeze(2).lu_solve(*H_LU).squeeze(2)

    S_ = torch.bmm(A_, invH_A_)
    S_LU = lu_hack(S_)
    t_ = torch.bmm(invH_g_.unsqueeze(1), A_.transpose(1, 2)).squeeze(1) - h_
    w_ = -t_.unsqueeze(2).lu_solve(*S_LU).squeeze(2)
    t_ = -g_ - w_.unsqueeze(1).bmm(A_).squeeze()
    v_ = t_.unsqueeze(2).lu_solve(*H_LU).squeeze(2)

    dx = v_[:, :nz]
    ds = v_[:, nz:]
    dz = w_[:, :nineq]
    dy = w_[:, nineq:] if neq > 0 else None

    return dx, ds, dz, dy


def solve_kkt(Q_LU, d, G, A, S_LU, rx, rs, rz, ry):
    """ Solve KKT equations for the affine step"""
    nineq, nz, neq, nBatch = get_sizes(G, A)

    invQ_rx = rx.unsqueeze(2).lu_solve(*Q_LU).squeeze(2)
    if neq > 0:
        h = torch.cat((invQ_rx.unsqueeze(1).bmm(A.transpose(1, 2)).squeeze(1) - ry,
                       invQ_rx.unsqueeze(1).bmm(G.transpose(1, 2)).squeeze(1) + rs / d - rz), 1)
    else:
        h = invQ_rx.unsqueeze(1).bmm(G.transpose(1, 2)).squeeze(1) + rs / d - rz

    w = -(h.unsqueeze(2).lu_solve(*S_LU)).squeeze(2)

    g1 = -rx - w[:, neq:].unsqueeze(1).bmm(G).squeeze(1)
    if neq > 0:
        g1 -= w[:, :neq].unsqueeze(1).bmm(A).squeeze(1)
    g2 = -rs - w[:, neq:]

    dx = g1.unsqueeze(2).lu_solve(*Q_LU).squeeze(2)
    ds = g2 / d
    dz = w[:, neq:]
    dy = w[:, :neq] if neq > 0 else None

    return dx, ds, dz, dy


def pre_factor_kkt(Q, G, A):
    """ Perform all one-time factorizations and cache relevant matrix products"""
    nineq, nz, neq, nBatch = get_sizes(G, A)

    try:
        Q_LU = lu_hack(Q)
    except:
        raise RuntimeError("""
qpth Error: Cannot perform LU factorization on Q.
Please make sure that your Q matrix is PSD and has
a non-zero diagonal.
""")

    # S = [ A Q^{-1} A^T        A Q^{-1} G^T          ]
    #     [ G Q^{-1} A^T        G Q^{-1} G^T + D^{-1} ]
    #
    # We compute a partial LU decomposition of the S matrix
    # that can be completed once D^{-1} is known.
    # See the 'Block LU factorization' part of our website
    # for more details.

    G_invQ_GT = torch.bmm(G, G.transpose(1, 2).lu_solve(*Q_LU))
    # G_invQ_GT = torch.bmm(G, G.transpose(1, 2).linalg.lu_solve(*Q_LU))

    R = G_invQ_GT.clone()
    S_LU_pivots = torch.IntTensor(range(1, 1 + neq + nineq)).unsqueeze(0) \
        .repeat(nBatch, 1).type_as(Q).int()
    if neq > 0:
        invQ_AT = A.transpose(1, 2).lu_solve(*Q_LU)
        A_invQ_AT = torch.bmm(A, invQ_AT)
        G_invQ_AT = torch.bmm(G, invQ_AT)

        LU_A_invQ_AT = lu_hack(A_invQ_AT)
        P_A_invQ_AT, L_A_invQ_AT, U_A_invQ_AT = torch.lu_unpack(*LU_A_invQ_AT)
        P_A_invQ_AT = P_A_invQ_AT.type_as(A_invQ_AT)

        S_LU_11 = LU_A_invQ_AT[0]
        U_A_invQ_AT_inv = (P_A_invQ_AT.bmm(L_A_invQ_AT)
                           ).lu_solve(*LU_A_invQ_AT)
        S_LU_21 = G_invQ_AT.bmm(U_A_invQ_AT_inv)
        T = G_invQ_AT.transpose(1, 2).lu_solve(*LU_A_invQ_AT)
        S_LU_12 = U_A_invQ_AT.bmm(T)
        S_LU_22 = torch.zeros(nBatch, nineq, nineq).type_as(Q)
        S_LU_data = torch.cat((torch.cat((S_LU_11, S_LU_12), 2),
                               torch.cat((S_LU_21, S_LU_22), 2)),
                              1)
        S_LU_pivots[:, :neq] = LU_A_invQ_AT[1]

        R -= G_invQ_AT.bmm(T)
    else:
        S_LU_data = torch.zeros(nBatch, nineq, nineq).type_as(Q)

    S_LU = [S_LU_data, S_LU_pivots]
    return Q_LU, S_LU, R


factor_kkt_eye = None


def factor_kkt(S_LU, R, d):
    """ Factor the U22 block that we can only do after we know D. """
    nBatch, nineq = d.size()
    neq = S_LU[1].size(1) - nineq
    # TODO: There's probably a better way to add a batched diagonal.
    global factor_kkt_eye
    if factor_kkt_eye is None or factor_kkt_eye.size() != d.size():
        # print('Updating batchedEye size.')
        factor_kkt_eye = torch.eye(nineq).repeat(
            nBatch, 1, 1).type_as(R).bool()
    T = R.clone()
    T[factor_kkt_eye] += (1. / d).squeeze().view(-1)

    T_LU = lu_hack(T)

    if not T.is_cuda:
        # TODO: Don't use pivoting in most cases because
        # torch.lu_unpack is inefficient here:
        oldPivotsPacked = S_LU[1][:, -nineq:] - neq
        oldPivots, _, _ = torch.lu_unpack(
            T_LU[0], oldPivotsPacked, unpack_data=False)
        newPivotsPacked = T_LU[1]
        newPivots, _, _ = torch.lu_unpack(
            T_LU[0], newPivotsPacked, unpack_data=False)

        # Re-pivot the S_LU_21 block.
        if neq > 0:
            S_LU_21 = S_LU[0][:, -nineq:, :neq]
            S_LU[0][:, -nineq:,
                    :neq] = newPivots.transpose(1, 2).bmm(oldPivots.bmm(S_LU_21))

        # Add the new S_LU_22 block pivots.
        S_LU[1][:, -nineq:] = newPivotsPacked + neq

    # Add the new S_LU_22 block.
    S_LU[0][:, -nineq:, -nineq:] = T_LU[0]
