import numpy as np
import torch
from enum import Enum
# from block import block


INACC_ERR = """
--------
qpth warning: Returning an inaccurate and potentially incorrect solutino.

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
    QR = 1


def forward(Qi, Qv, Qsz, p, Gi, Gv, Gsz, h, Ai, Av, Asz, b,
            eps=1e-12, verbose=0, notImprovedLim=3, maxIter=20):
    spTensor = torch.cuda.sparse.DoubleTensor if Qv.is_cuda else torch.sparse.DoubleTensor
    nBatch = Qv.size(0)
    Qs = [spTensor(Qi, Qv[j], Qsz) for j in range(nBatch)]
    Gs = [spTensor(Gi, Gv[j], Gsz) for j in range(nBatch)]
    As = [spTensor(Ai, Av[j], Asz) for j in range(nBatch)]

    nineq, nz = Gsz
    neq, _ = Asz

    solver = KKTSolvers.QR

    KKTeps = 1e-7  # For the regularized KKT matrix.

    # Find initial values
    if solver == KKTSolvers.QR:
        Di = torch.LongTensor([range(nineq), range(nineq)]).type_as(Qi)
        Dv = torch.ones(nBatch, nineq).type_as(Qv)
        Dsz = torch.Size([nineq, nineq])
        Ks, K, Didx = cat_kkt(Qi, Qv, Qsz, Gi, Gv, Gsz,
                              Ai, Av, Asz, Di, Dv, Dsz, 0.0)
        Ktildes, Ktilde, Didxtilde = cat_kkt(
            Qi, Qv, Qsz, Gi, Gv, Gsz, Ai, Av, Asz, Di, Dv, Dsz, KKTeps)
        assert torch.norm((Didx - Didxtilde).float()) == 0.0
        x, s, z, y = solve_kkt(Ks, K, Ktildes, Ktilde,
                               p, torch.zeros(nBatch, nineq).type_as(p),
                               -h, -b if b is not None else None)
    else:
        assert False

    M = torch.min(s, 1)[0].repeat(1, nineq)
    I = M < 0
    s[I] -= M[I] - 1

    M = torch.min(z, 1)[0].repeat(1, nineq)
    I = M < 0
    z[I] -= M[I] - 1

    best = {'resids': None, 'x': None, 'z': None, 's': None, 'y': None}
    nNotImproved = 0

    for i in range(maxIter):
        # affine scaling direction
        rx = torch.cat([(torch.mm(As[j].t(), y[j].unsqueeze(1)) if neq > 0 else 0.) +
                        torch.mm(Gs[j].t(), z[j].unsqueeze(1)) +
                        torch.mm(Qs[j], x[j].unsqueeze(1)) +
                        p[j] for j in range(nBatch)], 1).t()
        rs = z
        rz = torch.cat([torch.mm(Gs[j], x[j].unsqueeze(1)) +
                        s[j] - h[j] for j in range(nBatch)], 1).t()
        ry = torch.cat([torch.mm(As[j], x[j].unsqueeze(1)) - b[j]
                        for j in range(nBatch)], 1).t()
        mu = torch.abs((s * z).sum(1).squeeze() / nineq)
        z_resid = torch.norm(rz, 2, 1).squeeze()
        y_resid = torch.norm(ry, 2, 1).squeeze() if neq > 0 else 0
        pri_resid = y_resid + z_resid
        dual_resid = torch.norm(rx, 2, 1).squeeze()
        resids = pri_resid + dual_resid + nineq * mu

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

        if solver == KKTSolvers.QR:
            D = z / s
            K[1].t()[Didx] = D.t()
            Ktilde[1].t()[Didx] = D.t() + KKTeps
            # TODO: Share memory between these or handle batched sparse
            # matrices differently.
            for j in range(nBatch):
                Ks[j]._values()[Didx] = D[j]
                Ktildes[j]._values()[Didx] = D[j] + KKTeps
            dx_aff, ds_aff, dz_aff, dy_aff = solve_kkt(
                Ks, K, Ktildes, Ktilde, rx, rs, rz, ry)
        else:
            assert False

        # compute centering directions
        alpha = torch.min(torch.min(get_step(z, dz_aff),
                                    get_step(s, ds_aff)),
                          torch.ones(nBatch).type_as(Qv))
        alpha_nineq = alpha.repeat(nineq, 1).t()
        t1 = s + alpha_nineq * ds_aff
        t2 = z + alpha_nineq * dz_aff
        t3 = torch.sum(t1 * t2, 1).squeeze()
        t4 = torch.sum(s * z, 1).squeeze()
        sig = (t3 / t4)**3

        rx = torch.zeros(nBatch, nz).type_as(Qv)
        rs = ((-mu * sig).repeat(nineq, 1).t() + ds_aff * dz_aff) / s
        rz = torch.zeros(nBatch, nineq).type_as(Qv)
        ry = torch.zeros(nBatch, neq).type_as(Qv)

        if solver == KKTSolvers.QR:
            dx_cor, ds_cor, dz_cor, dy_cor = solve_kkt(
                Ks, K, Ktildes, Ktilde, rx, rs, rz, ry)
        else:
            assert False

        dx = dx_aff + dx_cor
        ds = ds_aff + ds_cor
        dz = dz_aff + dz_cor
        dy = dy_aff + dy_cor if neq > 0 else None
        alpha = torch.min(0.999 * torch.min(get_step(z, dz),
                                            get_step(s, ds)),
                          torch.ones(nBatch).type_as(Qv))

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
    # nBatch = v.size(0)
    a = -v / dv
    a[dv > 0] = max(1.0, a.max())
    return a.min(1)[0].squeeze()


def cat_kkt(Qi, Qv, Qsz, Gi, Gv, Gsz, Ai, Av, Asz, Di, Dv, Dsz, eps):
    nBatch = Qv.size(0)

    nineq, nz = Gsz
    neq, _ = Asz

    Di = Di + nz

    Gi_L = Gi.clone()
    Gi_L[0, :] += nz + nineq
    Gv_L = Gv

    Gi_U = torch.stack([Gi[1, :], Gi[0, :]])
    Gi_U[1, :] += nz + nineq
    Gv_U = Gv

    Ai_L = Ai.clone()
    Ai_L[0, :] += nz + 2 * nineq
    Av_L = Av

    Ai_U = torch.stack([Ai[1, :], Ai[0, :]])
    Ai_U[1, :] += nz + 2 * nineq
    Av_U = Av

    Ii_L = type(Qi)([range(nineq), range(nineq)])
    Ii_U = Ii_L.clone()
    Ii_L[0, :] += nz + nineq
    Ii_L[1, :] += nz
    Ii_U[0, :] += nz
    Ii_U[1, :] += nz + nineq
    Iv_L = type(Qv)(nBatch, nineq).fill_(1.0)
    Iv_U = Iv_L.clone()

    Ii_11 = type(Qi)([range(nz + nineq), range(nz + nineq)])
    Iv_11 = type(Qv)(nBatch, nz + nineq).fill_(eps)
    Ii_22 = type(Qi)([range(nz + nineq, nz + 2 * nineq + neq),
                      range(nz + nineq, nz + 2 * nineq + neq)])
    Iv_22 = type(Qv)(nBatch, nineq + neq).fill_(-eps)

    Ki = torch.cat((Qi, Di, Gi_L, Gi_U, Ai_L, Ai_U,
                    Ii_L, Ii_U, Ii_11, Ii_22), 1)
    Kv = torch.cat((Qv, Dv, Gv_L, Gv_U, Av_L, Av_U,
                    Iv_L, Iv_U, Iv_11, Iv_22), 1)
    k = nz + 2 * nineq + neq
    Ksz = torch.Size([k, k])

    I = torch.LongTensor(np.lexsort(
        (Ki[1].cpu().numpy(), Ki[0].cpu().numpy()))).cuda()
    Ki = Ki.t()[I].t().contiguous()
    Kv = Kv.t()[I].t().contiguous()

    Ks = [torch.cuda.sparse.DoubleTensor(
        Ki, Kv[i], Ksz).coalesce() for i in range(nBatch)]
    Ki = Ks[0]._indices()
    Kv = torch.stack([Ks[i]._values() for i in range(nBatch)])

    Didx = torch.nonzero(
        (Ki[0] == Ki[1]).__and__(nz <= Ki[0]).__and__(Ki[0] < nz + nineq)).squeeze()

    return Ks, [Ki, Kv, Ksz], Didx


def solve_kkt(Ks, K, Ktildes, Ktilde,
              rx, rs, rz, ry, niter=1):
    nBatch = len(Ks)
    nz = rx.size(1)
    nineq = rz.size(1)
    neq = ry.size(1)

    r = -torch.cat((rx, rs, rz, ry), 1)

    l = torch.spbqrfactsolve(*([r] + Ktilde))
    res = torch.stack([r[i] - torch.mm(Ks[i], l[i].unsqueeze(1))
                       for i in range(nBatch)])
    for k in range(niter):
        d = torch.spbqrfactsolve(*([res] + Ktilde))
        l = l + d
        res = torch.stack([r[i] - torch.mm(Ks[i], l[i].unsqueeze(1))
                           for i in range(nBatch)])

    solx = l[:, :nz]
    sols = l[:, nz:nz + nineq]
    solz = l[:, nz + nineq:nz + 2 * nineq]
    soly = l[:, nz + 2 * nineq:nz + 2 * nineq + neq]

    return solx, sols, solz, soly
