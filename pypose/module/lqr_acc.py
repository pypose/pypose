import torch
import torch.nn as nn
from .. import bmv, bvmv, cumops


class ACLQR(nn.Module):

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

    def forward(self, x_init):

        K, k = self.lqr_backward()
        tau = self.lqr_forward(x_init, K, k)
        return tau

    def lqr_backward(self):

        # Q: (B*, T, N, N), p: (B*, T, N), where B* can be any batch dimensions, e.g., (2, 3)
        B = self.p.shape[:-2]
        ns, nc = self.system.B.size(-2), self.system.B.size(-1)
        K = torch.zeros(B + (self.T, nc, ns), dtype=self.p.dtype, device=self.p.device)
        k = torch.zeros(B + (self.T, nc), dtype=self.p.dtype, device=self.p.device)

        for t in range(self.T-1, -1, -1):
            if t == self.T - 1:
                Qt = self.Q[...,t,:,:]
                qt = self.p[...,t,:]
            else:
                self.system.set_refpoint(t=t)
                F = torch.cat((self.system.A, self.system.B), dim=-1)
                Qt = self.Q[...,t,:,:] + F.mT @ V @ F
                qt = self.p[...,t,:] + bmv(F.mT, v)
                if self.system.c1 is not None:
                    qt = qt + bmv(F.mT @ V, self.system.c1)

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
        n_batch = self.p.size(-3)
        ns, nc = self.system.B.size(-2), self.system.B.size(-1)
        A = torch.zeros(n_batch, self.T, ns, ns, dtype=self.p.dtype, device=self.p.device)
        B = torch.zeros(n_batch, self.T, ns, nc, dtype=self.p.dtype, device=self.p.device)

        for t in range(self.T):
            self.system.set_refpoint(t=t)
            A[:,t,:,:] = self.system.A
            B[:,t,:,:] = self.system.B

        if A.ndim != 4:
            A = torch.tile(A.unsqueeze(-3), (1, self.T, 1, 1))
            B = torch.tile(B.unsqueeze(-3), (1, self.T, 1, 1))

        F = torch.cat((A, B), dim=-1)
        tau_0 = torch.cat((x_init, bmv(K[...,0,:,:], x_init) + k[...,0,:]), 1)
        N = torch.cat((torch.zeros(n_batch, self.T-1, ns, 1), k[...,1:,:].unsqueeze(-1)), dim=-2)
        KK = torch.cat((torch.zeros(n_batch, self.T-1, nc, nc), K[...,1:,:,:]), dim=-1)
        M = torch.cat((torch.zeros(n_batch, self.T-1, ns, ns+nc), KK), dim=-2)
        if self.system.c1 is not None and self.system.c1.ndim == 2:
            c1 = torch.tile(self.system.c1.unsqueeze(-2), (1, self.T-1, 1))
            N = N + torch.cat((c1.unsqueeze(-1), torch.zeros(n_batch, self.T-1, nc).unsqueeze(-1)), dim=-2) + \
                M @ torch.cat((torch.zeros(n_batch, self.T-1, nc).unsqueeze(-1), c1.unsqueeze(-1)), dim=-2)
        G = torch.cat((F, torch.zeros(n_batch, self.T, nc, ns+nc)), dim=-2)[...,:-1,:,:] + \
            M @ torch.cat((torch.zeros(n_batch, self.T, nc, ns+nc), F), dim=-2)[...,:-1,:,:]
        Gs = torch.flip(G, dims = [1])
        Gs = torch.cat((torch.tile(torch.eye(ns+nc)[None,:,:], (n_batch, 1, 1, 1)), Gs), dim=1)
        NN = torch.cat((torch.flip(N, dims = [1]), tau_0.unsqueeze(-1)[:,None,:,:]), dim=1)
        W = cumops(Gs, 1, lambda a, b : b @ a) @ NN
        tau = torch.sum(W, 1).mT.squeeze()

        return tau
