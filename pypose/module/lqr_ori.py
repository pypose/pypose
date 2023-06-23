import torch
import torch.nn as nn
from .. import bmv, bvmv

class LQRO(nn.Module):

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

    def forward(self, x_init, dt, current_x = None, current_u=None):

        K, k = self.lqr_backward(x_init, dt, current_x, current_u)
        x, u, cost = self.lqr_forward(x_init, K, k)

        return x, u, cost

    def lqr_backward(self, x_init, dt, current_x, current_u):

        ns, nsc = x_init.size(-1), self.p.size(-1)
        nc = nsc - ns

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
                if self.system.B.dim() > 2:
                    B = self.system.B.squeeze(-1)
                else:
                    B = self.system.B
                F = torch.cat((A, B), dim=-1)
                Qt = self.Q[...,t,:,:] + F.mT @ V @ F
                qt = p_new[...,t,:] + bmv(F.mT, v)

            Qxx, Qxu = Qt[..., :ns, :ns], Qt[..., :ns, ns:]
            Qux, Quu = Qt[..., ns:, :ns], Qt[..., ns:, ns:]
            qx, qu = qt[..., :ns], qt[..., ns:]
            Quu_inv = torch.linalg.pinv(Quu)
            K[...,t,:,:] = Kt = - Quu_inv @ Qux
            k[...,t,:] = kt = - bmv(Quu_inv, qu)
            V = Qxx + Qxu @ Kt + Kt.mT @ Qux + Kt.mT @ Quu @ Kt
            v = qx  + bmv(Qxu, kt) + bmv(Kt.mT, qu) + bmv(Kt.mT @ Quu, kt)

        return K, k

    def lqr_forward(self, x_init, K, k):

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

        for t in range(self.T):
            Kt, kt = K[...,t,:,:], k[...,t,:]
            delta_xt = xt - self.current_x[...,t,:]
            delta_u[..., t, :] = delta_ut = bmv(Kt, delta_xt) + kt
            u[...,t,:] = ut = delta_ut + self.current_u[...,t,:]
            xut = torch.cat((xt, ut), dim=-1)
            x[...,t+1,:] = xt = self.system(xt, ut)[0]
            cost = cost + 0.5 * bvmv(xut, self.Q[...,t,:,:], xut) + (xut * self.p[...,t,:]).sum(-1)

        return x[...,0:-1,:], u, cost
