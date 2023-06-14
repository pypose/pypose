import torch
import torch.nn as nn
from .. import bmv, bvmv

class LQR2(nn.Module):

    def __init__(self, system, Q, p, T):
        super().__init__()

        self.system = system
        self.Q, self.p, self.T = Q, p, T
        self.current_x = None
        self.current_u = None

        if self.Q.ndim == 3:
            self.Q = torch.tile(self.Q.unsqueeze(-3), (1, self.T, 1, 1))
        if self.p.ndim == 2:
            self.p = torch.tile(self.p.unsqueeze(-2), (1, self.T, 1))

        self.n_batch = self.p.shape[:-2]

        assert self.Q.shape[:-1] == self.p.shape, "Shape not compatible."
        assert self.Q.size(-1) == self.Q.size(-2), "Shape not compatible."
        assert self.Q.ndim == 4 or self.p.ndim == 3, "Shape not compatible."
        assert self.Q.device == self.p.device, "device not compatible."
        assert self.Q.dtype == self.p.dtype, "tensor data type not compatible."

    def forward(self, x_init, dt, current_x = None, current_u=None,
                u_lower=None, u_upper=None, du=None):

        K, k = self.lqr_backward(x_init, dt, current_x, current_u, u_lower, u_upper, du)
        x, u, cost = self.lqr_forward(x_init, K, k, u_lower, u_upper, du)

        return x, u, cost

    def lqr_backward(self, x_init, dt, current_x, current_u, u_lower, u_upper, du):

        ns, nsc = x_init.size(-1), self.p.size(-1)
        nc = nsc - ns
        prev_kt = None

        if current_u is None:
            current_u = torch.zeros(self.n_batch + (self.T, nc), device=self.p.device)
        if current_x is None:
            current_x = torch.zeros(self.n_batch + (self.T, ns), device=self.p.device)
            current_x[...,0,:] = x_init
            current_xt = x_init
            for i in range(self.T-1):
                current_x[...,i+1,:] = current_xt = self.system(current_xt, current_u[...,i,:])[0]

        self.current_x = current_x
        self.current_u = current_u

        K = torch.zeros(self.n_batch + (self.T, nc, ns), dtype=self.p.dtype, device=self.p.device)
        k = torch.zeros(self.n_batch + (self.T, nc), dtype=self.p.dtype, device=self.p.device)
        p_new = torch.zeros(self.n_batch + (self.T, nsc), dtype=self.p.dtype, device=self.p.device)

        for i in range(self.T):
            current_xut = torch.cat((current_x[...,i,:], current_u[...,i,:]), dim=-1)
            p_new[...,i,:] = bmv(self.Q[...,i,:,:], current_xut) + self.p[...,i,:]

        for t in range(self.T-1, -1, -1):
            if t == self.T - 1:
                Qt = self.Q[...,t,:,:]
                qt = p_new[...,t,:]
            else:
                self.system.set_refpoint(state=current_x[...,t,:], input=current_u[...,t,:],
                                         t=torch.tensor(t*dt))
                A = self.system.A.squeeze(-2)
                B = self.system.B.squeeze(-1)
                F = torch.cat((A, B), dim=-1)
                Qt = self.Q[...,t,:,:] + F.mT @ V @ F
                qt = p_new[...,t,:] + bmv(F.mT, v)

            Qxx, Qxu = Qt[..., :ns, :ns], Qt[..., :ns, ns:]
            Qux, Quu = Qt[..., ns:, :ns], Qt[..., ns:, ns:]
            qx, qu = qt[..., :ns], qt[..., ns:]
            Quu_inv = torch.linalg.pinv(Quu)

            if u_lower is None:
                K[...,t,:,:] = Kt = - Quu_inv @ Qux
                k[...,t,:] = kt = - bmv(Quu_inv, qu)
            else:
                lb = u_lower[...,t,:] - current_u[...,t,:]
                ub = u_upper[...,t,:] - current_u[...,t,:]

                if du is not None:
                    lb[lb < -du] = -du
                    ub[ub > du] = du

                kt, Quu_free_LU, If = pn(
                    Quu, qu, lb, ub, x_init=prev_kt, n_iter=20)

                prev_kt = kt
                Qux_ = Qux.clone()
                Qux_[(1-If).unsqueeze(2).repeat(1,1,Qux.size(2)).bool()] = 0


                if nc == 1:
                    K[...,t,:,:] = Kt = -((1./Quu_free_LU)*Qux_)
                else:
                    K[...,t,:,:] = Kt = -Qux_.lu_solve(*Quu_free_LU)

            V = Qxx + Qxu @ Kt + Kt.mT @ Qux + Kt.mT @ Quu @ Kt
            v = qx  + bmv(Qxu, kt) + bmv(Kt.mT, qu) + bmv(Kt.mT @ Quu, kt)

        return K, k

    def lqr_forward(self, x_init, K, k, u_lower, u_upper, du):

        assert x_init.device == K.device == k.device
        assert x_init.dtype == K.dtype == k.dtype
        assert x_init.ndim == 2, "Shape not compatible."

        ns, nc = self.current_x.size(-1), self.current_u.size(-1)
        u = torch.zeros(self.n_batch + (self.T, nc), dtype=self.p.dtype, device=self.p.device)
        delta_u = torch.zeros(self.n_batch + (self.T, nc), dtype=self.p.dtype, device=self.p.device)
        cost = torch.zeros(self.n_batch, dtype=self.p.dtype, device=self.p.device)
        x = torch.zeros(self.n_batch + (self.T+1, ns), dtype=self.p.dtype, device=self.p.device)
        x[..., 0, :] = x_init
        xt = x_init

        assert not ((delta_u is not None) and (u_lower is None))

        for t in range(self.T):
            Kt, kt = K[...,t,:,:], k[...,t,:]
            delta_xt = xt - self.current_x[...,t,:]
            delta_u[..., t, :] = delta_ut = bmv(Kt, delta_xt) + kt
            if u_lower is not None:
                lb = u_lower[...,t,:]
                ub = u_upper[...,t,:]
                if delta_u is not None:
                    lb_limit, ub_limit = lb, ub
                    lb = self.current_u[...,t,:] - du
                    ub = self.current_u[...,t,:] + du
                    I = lb < lb_limit
                    lb[I] = lb_limit if isinstance(lb_limit, float) else lb_limit[I]
                    I = ub > ub_limit
                    ub[I] = ub_limit if isinstance(lb_limit, float) else ub_limit[I]
                delta_u[..., t, :] = delta_ut = eclamp(delta_ut, lb, ub)

            u[...,t,:] = ut = delta_ut + self.current_u[...,t,:]
            xut = torch.cat((xt, ut), dim=-1)
            x[...,t+1,:] = xt = self.system(xt, ut)[0]
            cost = cost + 0.5 * bvmv(xut, self.Q[...,t,:,:], xut) + (xut * self.p[...,t,:]).sum(-1)

        return x[...,0:-1,:], u, cost


def pn(H, g, lb, up, x_init=None, n_iter=20):
    GAMMA = 0.1
    n_batch, n_ctrl, _ = H.size()
    pn_I = 1e-11*torch.eye(n_ctrl).type_as(H).expand_as(H)

    def obj(x):
        return 0.5 * bvmv(x, H, x) + (x * g).sum(-1)

    if x_init is None:
        if n_ctrl == 1:
            x_init = -(1./H.squeeze(2))*g
        else:
            H_lu = H.lu()
            x_init = -g.unsqueeze(2).lu_solve(*H_lu).squeeze(2)
    else:
        x_init = x_init.clone()

    x = eclamp(x_init, lb, up)

    J = torch.ones(n_batch).type_as(x).byte()

    for i in range(n_iter):
        g = bmv(H, x) + g

        Ic = (((x == lb) & (g > 0)) | ((x == up) & (g < 0))).float()
        If = 1-Ic

        if If.is_cuda:
            Hff_I = If.float().unsqueeze(2).bmm(If.float().unsqueeze(1)).type_as(If)
            not_Hff_I = 1-Hff_I
            Hfc_I = If.float().unsqueeze(2).bmm(Ic.float().unsqueeze(1)).type_as(If)
        else:
            Hff_I = If.unsqueeze(2).bmm(If.unsqueeze(1)).type_as(If)
            not_Hff_I = 1-Hff_I
            Hfc_I = If.unsqueeze(2).bmm(Ic.unsqueeze(1)).type_as(If)

        g_ = g.clone()
        g_[Ic.bool()] = 0.0
        H_ = H.clone()
        H_[not_Hff_I.bool()] = 0.0
        H_ = H_ + pn_I

        if n_ctrl == 1:
            dx = -(1./H_.squeeze(2))*g_
        else:
            H_lu_ = H_.lu()
            dx = -g_.unsqueeze(2).lu_solve(*H_lu_).squeeze(2)

        J = torch.norm(dx, 2, 1) >= 1e-4
        m = J.sum().item()
        if m == 0:
            return x, H_ if n_ctrl == 1 else H_lu_, If

        alpha = torch.ones(n_batch).type_as(x)
        decay = 0.1
        max_armijo = GAMMA
        count = 0

        while max_armijo <= GAMMA and count < 10:
            maybe_x = eclamp(x+torch.diag(alpha).mm(dx), lb, up)
            armijos = (GAMMA+1e-6)*torch.ones(n_batch).type_as(x)
            armijos[J] = (obj(x)-obj(maybe_x))[J]/(torch.bmm(g.unsqueeze(1), (x-maybe_x).unsqueeze(2)).squeeze(1).squeeze(1))[J]
            I = armijos <= GAMMA
            alpha[I] *= decay
            max_armijo = torch.max(armijos)
            count += 1

        x = maybe_x

    return x, H_ if n_ctrl == 1 else H_lu_, If


def eclamp(x, lb, ub):

    I = x < lb
    x[I] = lb[I]

    I = x > ub
    x[I] = ub[I]

    return x
