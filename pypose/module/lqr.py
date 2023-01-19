import torch
import torch.nn as nn
from ..basics import bmv, bvmv


class LQR(nn.Module):
    r'''
    Linear Quadratic Regulator (LQR) with Dynamic Programming.

    Args:
        system (instance): The system to be soved by LQR.

    LQR finds the optimal nominal trajectory :math:`\mathbf{\tau}_{1:T}^*` = 
    :math:`\begin{Bmatrix} \mathbf{x}_t, \mathbf{u}_t \end{Bmatrix}_{1:T}` 
    of the optimization problem

    .. math::
        \begin{aligned}
            \mathbf{\tau}_{1:T}^* = \mathop{\arg\min}\limits_{\tau_{1:T}} \sum\limits_t\frac{1}{2}
            \mathbf{\tau}_t^\top\mathbf{Q}_t\mathbf{\tau}_t + \mathbf{p}_t^\top\mathbf{\tau}_t \\
            \mathrm{s.t.} \quad \mathbf{x}_1 = \mathbf{x}_{init}, 
            \ \mathbf{x}_{t+1} = \mathbf{F}_t\mathbf{\tau}_t + \mathbf{f}_t \\
        \end{aligned}

    where :math:`\mathbf{\tau}` = :math:`\begin{bmatrix} \mathbf{x} & \mathbf{u} \end{bmatrix}^\top`, 
    :math:`\mathbf{F}` = :math:`\begin{bmatrix} \mathbf{A} & \mathbf{B} \end{bmatrix}`, 
    :math:`\mathbf{f}` = :math:`\mathbf{c}_1`.

    From a policy learning perspective, this can be interpreted as a module with unknown parameters
    :math:`\begin{Bmatrix} \mathbf{Q}, \mathbf{p}, \mathbf{F}, \mathbf{f} \end{Bmatrix}`, 
    which can be integrated into a larger end-to-end learning system.

    Note:
        Here, we consider the system sent to LQR as a linear system in each small horizon. 
        There are many ways of solving the LQR problem, such as solving Riccati equation, 
        dynamic programming (DP), etc. Here, we implement the DP method.
        
        For more details about mathematical process, please refer to 
        http://rll.berkeley.edu/deeprlcourse/f17docs/lecture_8_model_based_planning.pdf

    Example:
        >>> n_batch = 2
        >>> ns, nc = 4, 3
        >>> n_sc = ns + nc
        >>> T = 5
        >>> Q = torch.randn(n_batch, T, n_sc, n_sc)
        >>> Q = torch.matmul(Q.mT, Q)
        >>> p = torch.randn(n_batch, T, n_sc)
        >>> A = torch.tile(torch.eye(ns) + 0.2 * torch.randn(ns, ns), (n_batch, 1, 1))
        >>> B = torch.randn(n_batch, ns, nc)
        >>> C = torch.tile(torch.eye(ns), (n_batch, 1, 1))
        >>> D = torch.tile(torch.zeros(ns, nc), (n_batch, 1, 1))
        >>> c1 = torch.tile(torch.randn(ns), (n_batch, 1))
        >>> c2 = torch.tile(torch.zeros(ns), (n_batch, 1))
        >>> x_init = torch.randn(n_batch, ns)
        >>> lti = pp.module.LTI(A, B, C, D, c1, c2)
        >>> lqr = pp.module.LQR(lti)
        >>> x_lqr, u_lqr, objs_lqr, tau = lqr(x_init, Q, p)
        >>> print(x_lqr)
        >>> print(u_lqr)
        tensor([[[-0.5453, -0.5922, -1.2467,  0.8291],
                 [ 0.2576, -0.5661, -0.9135,  1.1018]],
                [[-2.9563,  3.2528,  0.6299, -0.5684],
                 [[ 0.9996,  1.7761, -1.6918,  2.2817]],
                [[ 1.7806,  7.9820,  4.4041, -4.8167],
                 [[ 2.2647,  5.1426, -0.9377,  0.7258]],
                [[-7.0567,  2.4997, -3.3157,  3.5961],
                 [[-4.1525,  5.0354, -2.7758,  4.3140]],
                [[-5.1267,  6.8131, -0.6195,  3.7960],
                 [[-0.2206,  9.1061, -0.0106,  3.7945]]]
        tensor([[[-4.1313,  6.6788, -2.7602],
                 [ 0.0752,  1.1968, -0.2460]],
                [[ 2.6432,  1.6548, -3.2089],
                 [-1.7543,  4.5867, -4.6340]],
                [[-3.4785, -3.5352,  1.6931],
                 [-6.5703,  5.4751, -3.8288]],
                [[-0.2215,  4.0535, -3.0925],
                 [ 0.5750,  3.5659, -5.1008]],
                [[ 4.9618,  6.6287, -7.5575],
                 [ 0.3090, -3.1843,  0.7678]]] 
    '''
    def __init__(self, system):
        super().__init__()
        self.system = system

    def forward(self, x, Q, p):
        r'''
        Perform one step advance for the LQR problem.
        '''
        K, k = self.lqr_backward(Q, p)
        x, u, cost = self.lqr_forward(x, Q, p, K, k)
        return x, u, cost

    def lqr_backward(self, Q, p):
        r'''
        The backward recursion of the dynamic programming to solve LQR.
        '''
        # Q: (B*, T, N, N), p: (B*, T, N), where B* can be any batch dimensions, e.g., (2, 3)
        assert Q.shape[:-1] == p.shape and Q.size(-1) == Q.size(-2), "Shape not compatible."
        B, T = p.shape[:-2], p.size(-2)
        ns, nc = self.system.B.size(-2), self.system.B.size(-1)
        K = torch.zeros(B + (T, nc, ns), dtype=p.dtype, device=p.device)
        k = torch.zeros(B + (T, nc), dtype=p.dtype, device=p.device)

        for t in range(T-1, -1, -1): 
            if t == T - 1:
                Qt = Q[...,t,:,:]
                qt = p[...,t,:]
            else:
                self.system.set_refpoint(t=t)
                F = torch.cat((self.system.A, self.system.B), dim=-1)
                Qt = Q[...,t,:,:] + F.mT @ V @ F
                qt = p[...,t,:] + bmv(F.mT, v)
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

    def lqr_forward(self, x_init, Q, p, K, k):
        r'''
        The forward recursion of the dynamic programming to solve LQR.
        '''
        assert x_init.device == Q.device == p.device == K.device == k.device
        assert x_init.dtype == Q.dtype == p.dtype == K.dtype == k.dtype
        B, T, nc = p.shape[:-2], p.size(-2), self.system.B.size(-1)
        u = torch.zeros(B + (T, nc), dtype=p.dtype, device=p.device)
        cost = torch.zeros(B, dtype=p.dtype, device=p.device)
        x = x_init.repeat(B + (T+1, 1))

        for t in range(T):
            Kt, xt, kt = K[...,t,:,:], x[...,t,:], k[...,t,:]
            u[..., t, :] = ut = bmv(Kt, xt) + kt
            xut = torch.cat((xt, ut), dim=-1)
            self.system.set_refpoint(t=t)
            x[...,t+1,:] = self.system(xt, ut)[0]
            cost = cost + 0.5 * bvmv(xut, Q[...,t,:,:], xut) + (xut * p[...,t,:]).sum(-1)

        return x[...,0:,:], u, cost
