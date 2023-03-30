import torch
import torch.nn as nn
from ..basics import bmv, bvmv


class NLQR(nn.Module):

    def __init__(self, system, Q, p, T):
        super().__init__()
        self.system = system
        self.Q, self.p, self.T = Q, p, T

        if self.Q.ndim == 3:
            self.Q = torch.tile(self.Q.unsqueeze(-3), (1, self.T, 1, 1))

        if self.p.ndim == 2:
            self.p = torch.tile(self.p.unsqueeze(-2), (1, self.T, 1))

        assert self.Q.shape[:-1] == self.p.shape, "Shape not compatible."
        assert self.Q.size(-1) == self.Q.size(-2), "Shape not compatible."
        assert self.Q.ndim == 4 or self.p.ndim == 3, "Shape not compatible."
        assert self.Q.device == self.p.device, "device not compatible."
        assert self.Q.dtype == self.p.dtype, "tensor data type not compatible."

    def forward(self, x_init, current_x, current_u, time):

        K, k = self.lqr_backward(current_x, current_u, time)
        x, u, cost = self.lqr_forward(x_init, current_x, current_u, K, k)
        return x, u, cost

    def lqr_backward(self, current_x, current_u, time):

        # Q: (B*, T, N, N), p: (B*, T, N), where B* can be any batch dimensions, e.g., (2, 3)
        n_batch = self.p.shape[:-2] 
        ns, nc = current_x.size(-1), current_u.size(-1)
        K = torch.zeros(n_batch + (self.T, nc, ns), dtype=self.p.dtype, device=self.p.device)
        k = torch.zeros(n_batch + (self.T, nc), dtype=self.p.dtype, device=self.p.device)
        c1 = None

        A = torch.zeros(n_batch + (self.T, ns, ns), dtype=self.p.dtype, device=self.p.device)
        B = torch.zeros(n_batch + (self.T, ns, nc), dtype=self.p.dtype, device=self.p.device)
        for t in range(self.T):
            self.system.set_refpoint(state=current_x[...,t,:], input=current_u[...,t,:], t=time[t])
            A[...,t,:,:] = self.system.A.squeeze(-2)
            B[...,t,:,:] = self.system.B.squeeze(-1) 

        for t in range(self.T-1, -1, -1): 
            if t == self.T - 1:
                Qt = self.Q[...,t,:,:]
                qt = self.p[...,t,:]
            else:
                F = torch.cat((A[...,t,:,:], B[...,t,:,:]), dim=-1)
                Qt = self.Q[...,t,:,:] + F.mT @ V @ F
                qt = self.p[...,t,:] + bmv(F.mT, v)
                if c1 is not None:
                    qt = qt + bmv(F.mT @ V, c1)

            Qxx, Qxu = Qt[..., :ns, :ns], Qt[..., :ns, ns:]
            Qux, Quu = Qt[..., ns:, :ns], Qt[..., ns:, ns:]
            qx, qu = qt[..., :ns], qt[..., ns:]
            Quu_inv = torch.linalg.pinv(Quu)

            K[...,t,:,:] = Kt = - Quu_inv @ Qux
            k[...,t,:] = kt = - bmv(Quu_inv, qu)
            V = Qxx + Qxu @ Kt + Kt.mT @ Qux + Kt.mT @ Quu @ Kt
            v = qx + bmv(Qxu, kt) + bmv(Kt.mT, qu) + bmv(Kt.mT @ Quu, kt)

        return K, k

    def lqr_forward(self, x_init, current_x, current_u, K, k):

        assert x_init.device == K.device == k.device 
        assert x_init.dtype == K.dtype == k.dtype
        assert x_init.ndim == 2, "Shape not compatible."
        n_batch = self.p.shape[:-2]
        ns, nc = current_x.size(-1), current_u.size(-1)
        u = torch.zeros(n_batch + (self.T, nc), dtype=self.p.dtype, device=self.p.device)
        delta_u = torch.zeros(n_batch + (self.T, nc), dtype=self.p.dtype, device=self.p.device)
        cost = torch.zeros(n_batch, dtype=self.p.dtype, device=self.p.device)
        delta_xt = torch.zeros_like(x_init, dtype=x_init.dtype, device=self.p.device)
        x = torch.zeros(n_batch + (self.T+1, ns), dtype=self.p.dtype, device=self.p.device)
        x[..., 0, :] = x_init
        xt = x_init

        for t in range(self.T):
            Kt, kt = K[...,t,:,:], k[...,t,:]
            delta_ut = bmv(Kt, delta_xt) + kt
            delta_u[..., t, :] = delta_ut.clone()
            current_ut = current_u[...,t,:].clone()
            ut = delta_ut + current_ut
            u[...,t,:] = ut.clone()
            xut = torch.cat((xt, ut), dim=-1)
            x[...,t+1,:] = self.system(xt, ut)[0]
            xt = x[...,t+1,:].clone()
            cost = cost + 0.5 * bvmv(xut, self.Q[...,t,:,:], xut) + (xut * self.p[...,t,:]).sum(-1)

        return x[...,0:-1,:], u, cost
    