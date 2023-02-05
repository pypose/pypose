import torch
import torch.nn as nn
from ..basics import bmv, bvmv
import time


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
    where :math:`\mathbf{\tau}` = :math:`\begin{bmatrix} \mathbf{x} \\ \mathbf{u} \end{bmatrix}`, 
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
        >>> n_batch, T = 2, 5
        >>> n_state, n_ctrl = 4, 3
        >>> n_sc = n_state + n_ctrl
        >>> Q = torch.randn(n_batch, T, n_sc, n_sc)
        >>> Q = torch.matmul(Q.mT, Q)
        >>> p = torch.randn(n_batch, T, n_sc)
        >>> A = torch.tile(torch.eye(n_state) + 0.2 * torch.randn(n_state, n_state), (n_batch, 1, 1))
        >>> B = torch.randn(n_batch, n_state, n_ctrl)
        >>> C = torch.tile(torch.eye(n_state), (n_batch, 1, 1))
        >>> D = torch.tile(torch.zeros(n_state, n_ctrl), (n_batch, 1, 1))
        >>> c1 = torch.tile(torch.randn(n_state), (n_batch, 1))
        >>> c2 = torch.tile(torch.zeros(n_state), (n_batch, 1))
        >>> x_init = torch.randn(n_batch, n_state)
        >>> lti = pp.module.LTI(A, B, C, D, c1, c2)
        >>> LQR = pp.module.LQR(lti)
        >>> x, u, cost = LQR(x_init, Q, p)
        >>> print("x = ", x)
        >>> print("u = ", u)
        x =  tensor([[[-0.2633, -0.3466,  2.3803, -0.0423],
                      [ 0.1849, -1.3884,  1.0898, -1.6229],
                      [ 1.2138, -0.7161,  0.2954, -0.6819],
                      [ 1.4840, -1.1249, -1.0302,  0.9805],
                      [-0.3477, -1.7063,  4.6494,  2.6780]],

                     [[-0.9744,  0.4976,  0.0603, -0.5258],
                      [-0.6356,  0.0539,  0.7264, -0.5048],
                      [-0.2275, -0.1649,  0.3872, -0.4614],
                      [ 0.2697, -0.3576,  0.0999, -0.4594],
                      [ 0.3916, -2.0832,  0.0701, -0.5407]]])
        
        u =  tensor([[[ 1.0405,  0.1586, -0.1282],
                      [-1.4845, -0.5745,  0.2523],
                      [-0.6322, -0.3281, -0.3620],
                      [-1.6768,  2.4054, -0.1047],
                      [-1.7948,  3.5269,  9.0703]],

                     [[-0.1795,  0.9153,  1.7066],
                      [ 0.0814,  0.4004,  0.7114],
                      [ 0.0435,  0.5782,  1.0127],
                      [-0.3017, -0.2897,  0.7251],
                      [-0.0728,  0.7290, -0.3117]]])

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
        for :math:`\t` = :math:`\T` to 1:
        .. math::
            \begin{aligned}
                \mathbf{Q}_t = \mathbf{C}_t + \mathbf{F}_t^\top\mathbf{V}_t+1\mathbf{F}_t \\
                \mathbf{q}_t = \mathbf{c}_t + \mathbf{F}_t^\top\mathbf{V}_t+1\mathbf{f}_t + \mathbf{F}_t^\top\mathbf{v}_t+1 \\
                \mathbf{K}_t = -\mathbf{Q}_{\mathbf{u}_t, \mathbf{u}_t}^{-1}\mathbf{Q}_{\mathbf{u}_t, \mathbf{x}_t} \\
                \mathbf{k}_t = -\mathbf{Q}_{\mathbf{u}_t, \mathbf{u}_t}^{-1}\mathbf{q}_{\mathbf{u}_t} \\
                \mathbf{V}_t = \mathbf{Q}_{\mathbf{x}_t, \mathbf{x}_t} + 
                \mathbf{Q}_{\mathbf{x}_t, \mathbf{u}_t}\mathbf{K}_t + 
                \mathbf{K}_t^\top\mathbf{Q}_{\mathbf{u}_t, \mathbf{x}_t} + 
                \mathbf{K}_t^\top\mathbf{Q}_{\mathbf{u}_t, \mathbf{u}_t}\mathbf{K}_t \\
                \mathbf{v}_t = \mathbf{q}_{\mathbf{x}_t} + 
                \mathbf{Q}_{\mathbf{x}_t, \mathbf{u}_t}\mathbf{k}_t + 
                \mathbf{K}_t^\top\mathbf{q}_{\mathbf{u}_t} + 
                \mathbf{K}_t^\top\mathbf{Q}_{\mathbf{u}_t, \mathbf{u}_t}\mathbf{k}_t \\
            \end{aligned}
        
        Args:
            Q (:obj:`Tensor`): The matrix of quadratic parameter.
            p (:obj:`Tensor`): The constant vector of quadratic parameter.

        Returns:
            Tuple of Tensor: The status feedback controller :math:`\mathbf{K}` and 
            :math:`\mathbf{k}` at all steps.
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
        for :math:`\t` = 1 to :math:`\T`:
        .. math::
            \begin{aligned}
                \mathbf{u}_t = \mathbf{K}_t\mathbf{x}_t + \mathbf{k}_t \\
                \mathbf{x}_t+1 = f \left( \mathbf{x}_t, \mathbf{u}_t \right) \\
            \end{aligned}
        where :math:`f \left( \mathbf{x}_t, \mathbf{u}_t \right)` represents the system dynamics.

        Args:
            Q (:obj:`Tensor`): The matrix of quadratic parameter.
            p (:obj:`Tensor`): The constant of quadratic parameter.
            K (:obj:`Tensor`): The matrix of status feedback controller at all steps.
            k (:obj:`Tensor`): The constant of status feedback controller at all steps.

        Returns:
            Tuple of Tensor: The state of the dynamical system :math:`\mathbf{x}`, 
            the input to the dynamical system :math:`\mathbf{u}`,
            and the costs to the dynamical system :math:`\mathbf{c}`.
        '''
        t1 = time.time()
        assert x_init.device == Q.device == p.device == K.device == k.device
        assert x_init.dtype == Q.dtype == p.dtype == K.dtype == k.dtype
        B, T, ns, nc = p.shape[:-2], p.size(-2), self.system.B.size(-2), self.system.B.size(-1)
        u = torch.zeros(B + (T, nc), dtype=p.dtype, device=p.device)
        cost = torch.zeros(B, dtype=p.dtype, device=p.device)
        x = torch.zeros(B + (T+1, ns), dtype=p.dtype, device=p.device)
        x[..., 0, :] = x_init

        self.system.set_refpoint(t=0)
        for t in range(T):
            Kt, xt, kt = K[...,t,:,:], x[...,t,:], k[...,t,:]
            u[..., t, :] = ut = bmv(Kt, xt) + kt
            xut = torch.cat((xt, ut), dim=-1)
            x[...,t+1,:] = self.system(xt, ut)[0]
            cost = cost + 0.5 * bvmv(xut, Q[...,t,:,:], xut) + (xut * p[...,t,:]).sum(-1)

        t2 = time.time()
        print(t2-t1)

        return x[...,0:-1,:], u, cost