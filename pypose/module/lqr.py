import torch
import torch.nn as nn
from .. import bmv, bvmv


class LQR(nn.Module):
    r'''
    Linear Quadratic Regulator (LQR) with Dynamic Programming.

    Args:
        system (:obj:`instance`): The system to be soved by LQR.
        Q (:obj:`Tensor`): The weight matrix of the quadratic term.
        p (:obj:`Tensor`): The weight vector of the first-order term.
        T (:obj:`int`): Time steps of system.

    A discrete-time linear system can be described as:

    .. math::
        \begin{align*}
            \mathbf{x}_{t+1} &= \mathbf{A}_t\mathbf{x}_t + \mathbf{B}_t\mathbf{u}_t
                                                         + \mathbf{c}_{1t}          \\
            \mathbf{y}_t &= \mathbf{C}_t\mathbf{x}_t + \mathbf{D}_t\mathbf{u}_t
                                                     + \mathbf{c}_{2t}              \\
        \end{align*}

    where :math:`\mathbf{x}`, :math:`\mathbf{u}` are the state and input of the linear system;
    :math:`\mathbf{y}` is the observation of the linear system; :math:`\mathbf{A}`,
    :math:`\mathbf{B}` are the state matrix and input matrix of the linear system;
    :math:`\mathbf{C}`, :math:`\mathbf{D}` are the output matrix and observation matrix of the
    linear system; :math:`\mathbf{c}_{1}`, :math:`\mathbf{c}_{2}` are the constant input and
    constant output of the linear system. The subscript :math:`\cdot_{t}` denotes the time step.

    LQR finds the optimal nominal trajectory :math:`\mathbf{\tau}_{1:T}^*` =
    :math:`\begin{Bmatrix} \mathbf{x}_t, \mathbf{u}_t \end{Bmatrix}_{1:T}`
    for the linear system of the optimization problem:

    .. math::
        \begin{align*}
          \mathbf{\tau}_{1:T}^* = \mathop{\arg\min}\limits_{\tau_{1:T}} \sum\limits_t\frac{1}{2}
          \mathbf{\tau}_t^\top\mathbf{Q}_t\mathbf{\tau}_t + \mathbf{p}_t^\top\mathbf{\tau}_t \\
          \mathrm{s.t.} \quad \mathbf{x}_1 = \mathbf{x}_{\text{init}}, \\
          \mathbf{x}_{t+1} = \mathbf{F}_t\mathbf{\tau}_t + \mathbf{c}_{1t} \\
        \end{align*}

    where :math:`\mathbf{\tau}_t` = :math:`\begin{bmatrix} \mathbf{x}_t \\ \mathbf{u}_t
    \end{bmatrix}`, :math:`\mathbf{F}_t` = :math:`\begin{bmatrix} \mathbf{A}_t & \mathbf{B}_t
    \end{bmatrix}`.

    The LQR process can be summarised as a backward and a forward recursion.

    - The backward recursion.

      For :math:`t` = :math:`T` to 1:

        .. math::
            \begin{align*}
                \mathbf{Q}_t &= \mathbf{Q}_t + \mathbf{F}_t^\top\mathbf{V}_{t+1}\mathbf{F}_t \\
                \mathbf{q}_t &= \mathbf{q}_t + \mathbf{F}_t^\top\mathbf{V}_{t+1}
                                        \mathbf{c}_{1t} + \mathbf{F}_t^\top\mathbf{v}_{t+1}  \\
                \mathbf{K}_t &= -\mathbf{Q}_{\mathbf{u}_t, \mathbf{u}_t}^{-1}
                                                     \mathbf{Q}_{\mathbf{u}_t, \mathbf{x}_t} \\
                \mathbf{k}_t &= -\mathbf{Q}_{\mathbf{u}_t, \mathbf{u}_t}^{-1}
                                                           \mathbf{q}_{\mathbf{u}_t}         \\
                \mathbf{V}_t &= \mathbf{Q}_{\mathbf{x}_t, \mathbf{x}_t}
                    + \mathbf{Q}_{\mathbf{x}_t, \mathbf{u}_t}\mathbf{K}_t
                    + \mathbf{K}_t^\top\mathbf{Q}_{\mathbf{u}_t, \mathbf{x}_t}
                    + \mathbf{K}_t^\top\mathbf{Q}_{\mathbf{u}_t, \mathbf{u}_t}\mathbf{K}_t   \\
                \mathbf{v}_t &= \mathbf{q}_{\mathbf{x}_t}
                    + \mathbf{Q}_{\mathbf{x}_t, \mathbf{u}_t}\mathbf{k}_t
                    + \mathbf{K}_t^\top\mathbf{q}_{\mathbf{u}_t}
                    + \mathbf{K}_t^\top\mathbf{Q}_{\mathbf{u}_t, \mathbf{u}_t}\mathbf{k}_t   \\
            \end{align*}

    - The forward recursion.

      For :math:`t` = 1 to :math:`T`:

        .. math::
            \begin{align*}
                \mathbf{u}_t &= \mathbf{K}_t\mathbf{x}_t + \mathbf{k}_t \\
                \mathbf{x}_{t+1} &= \mathbf{A}_t\mathbf{x}_t + \mathbf{B}_t\mathbf{u}_t
                                                             + \mathbf{c}_{1t} \\
            \end{align*}

    Then quadratic costs of the system over the time horizon:

        .. math::
            \mathbf{c} \left( \mathbf{\tau}_t \right) = \frac{1}{2}
            \mathbf{\tau}_t^\top\mathbf{Q}_t\mathbf{\tau}_t + \mathbf{p}_t^\top\mathbf{\tau}_t

    A discrete-time non-linear system can be described as:

    .. math::
        \begin{aligned}
            \mathbf{x}_{t+1} &= \mathbf{f}(\mathbf{x}_t, \mathbf{u}_t, t_t) \\
            \mathbf{y}_{t} &= \mathbf{g}(\mathbf{x}_t, \mathbf{u}_t, t_t) \\
        \end{aligned}

    For a current point :math:`\chi^*=(\mathbf{x}^*, \mathbf{u}^*, t^*)` along a
    trajectory, consider a point :math:`\chi=(\mathbf{x}^*+\delta\mathbf{x}, \mathbf{u}^*
    +\delta\mathbf{u}, t^*)` near :math:`\chi^*`. We could do a linear approximation:

    .. math::
            \begin{aligned}
            \mathbf{f}(\mathbf{x}, \mathbf{u}, t^*) &\approx \mathbf{f}(\mathbf{x}^*,
                \mathbf{u}^*, t^*) +  \left. \frac{\partial \mathbf{f}}{\partial \mathbf{x}}
                \right|_{\chi^*} \delta \mathbf{x} + \left. \frac{\partial \mathbf{f}}
                {\partial \mathbf{u}} \right|_{\chi^*} \delta \mathbf{u} \\
            &= \mathbf{f}(\mathbf{x}^*, \mathbf{u}^*, t^*) + \mathbf{A} \delta \mathbf{x}
                + \mathbf{B} \delta \mathbf{u} \\
            \delta \mathbf{x}_{t+1} &= \mathbf{A}_t \delta \mathbf{x}_t + \mathbf{B}_t
                \delta \mathbf{u}_t \\
            &= \mathbf{F}_t \delta \mathbf{\tau}_t \\
            \end{aligned}

    where :math:`\delta \mathbf{\tau}_t` = :math:`\begin{bmatrix} \delta \mathbf{x}_t \\
    \delta \mathbf{u}_t \end{bmatrix}`, :math:`\mathbf{F}_t` = :math:`\begin{bmatrix}
    \mathbf{A}_t & \mathbf{B}_t \end{bmatrix}`.

    Now we can run LQR with :math:`\delta \mathbf{x}_t` and :math:`\delta \mathbf{x}_t`.

    - The backward recursion.

      For :math:`t` = :math:`T` to 1:

        .. math::
            \begin{align*}
                \mathbf{Q}_t &= \mathbf{Q}_t + \mathbf{F}_t^\top\mathbf{V}_{t+1}\mathbf{F}_t \\
                \mathbf{q}_t &= \mathbf{q}_t + \mathbf{F}_t^\top\mathbf{v}_{t+1}  \\
                \mathbf{K}_t &= -\mathbf{Q}_{\delta \mathbf{u}_t, \delta \mathbf{u}_t}^{-1}
                                    \mathbf{Q}_{\delta \mathbf{u}_t, \delta \mathbf{x}_t} \\
                \mathbf{k}_t &= -\mathbf{Q}_{\delta \mathbf{u}_t, \delta \mathbf{u}_t}^{-1}
                                    \mathbf{q}_{\delta \mathbf{u}_t}         \\
                \mathbf{V}_t &= \mathbf{Q}_{\delta \mathbf{x}_t, \delta \mathbf{x}_t}
                    + \mathbf{Q}_{\delta \mathbf{x}_t, \delta \mathbf{u}_t}\mathbf{K}_t
                    + \mathbf{K}_t^\top\mathbf{Q}_{\delta \mathbf{u}_t, \delta \mathbf{x}_t}
                    + \mathbf{K}_t^\top\mathbf{Q}_{\delta \mathbf{u}_t, \delta \mathbf{u}_t}
                        \mathbf{K}_t   \\
                \mathbf{v}_t &= \mathbf{q}_{\delta \mathbf{x}_t}
                    + \mathbf{Q}_{\delta \mathbf{x}_t, \delta \mathbf{u}_t}\mathbf{k}_t
                    + \mathbf{K}_t^\top\mathbf{q}_{\delta \mathbf{u}_t}
                    + \mathbf{K}_t^\top\mathbf{Q}_{\delta \mathbf{u}_t,
                        \delta \mathbf{u}_t}\mathbf{k}_t   \\
            \end{align*}

    - The forward recursion.

      For :math:`t` = 1 to :math:`T`:

        .. math::
            \begin{align*}
                \delta \mathbf{u}_t &= \mathbf{K}_t \delta \mathbf{x}_t + \mathbf{k}_t \\
                \mathbf{u}_t &= \delta \mathbf{u}_t + \mathbf{u}_t^* \\
                \mathbf{x}_{t+1} &= \mathbf{f}(\mathbf{x}_t, \mathbf{u}_t) \\
            \end{align*}

    Then quadratic costs of the system over the time horizon:

        .. math::
            \mathbf{c} \left( \mathbf{\tau}_t \right) = \frac{1}{2}
            \mathbf{\tau}_t^\top\mathbf{Q}_t\mathbf{\tau}_t + \mathbf{p}_t^\top\mathbf{\tau}_t

    Note:
        The discrete-time system to be solved by LQR could be both linear time-invariant
        (:meth:`LTI`) system, linear time-varying (:meth:`LTV`) system or non-linear
        (:meth:`NLS`) system.

    From the learning perspective, this can be interpreted as a module with unknown parameters
    :math:`\begin{Bmatrix} \mathbf{Q}, \mathbf{p}, \mathbf{F}, \mathbf{f} \end{Bmatrix}`,
    which can be integrated into a larger end-to-end learning system.

    Note:
        The implementation is based on page 24-32 of `Optimal Control and Planning
        <http://rll.berkeley.edu/deeprlcourse/f17docs/lecture_8_model_based_planning.pdf>`_.

    Example:
        >>> n_batch, T = 2, 5
        >>> n_state, n_ctrl = 4, 3
        >>> n_sc = n_state + n_ctrl
        >>> Q = torch.randn(n_batch, T, n_sc, n_sc)
        >>> Q = torch.matmul(Q.mT, Q)
        >>> p = torch.randn(n_batch, T, n_sc)
        >>> r = 0.2 * torch.randn(n_state, n_state)
        >>> A = torch.tile(torch.eye(n_state) + r, (n_batch, 1, 1))
        >>> B = torch.randn(n_batch, n_state, n_ctrl)
        >>> C = torch.tile(torch.eye(n_state), (n_batch, 1, 1))
        >>> D = torch.tile(torch.zeros(n_state, n_ctrl), (n_batch, 1, 1))
        >>> c1 = torch.tile(torch.randn(n_state), (n_batch, 1))
        >>> c2 = torch.tile(torch.zeros(n_state), (n_batch, 1))
        >>> x_init = torch.randn(n_batch, n_state)
        >>>
        >>> lti = pp.module.LTI(A, B, C, D, c1, c2)
        >>> LQR = pp.module.LQR(lti, Q, p, T)
        >>> x, u, cost = LQR(x_init)
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

    def forward(self, x_init, current_x=None, current_u=None, time=None):
        r'''
        Performs LQR for the linear system.

        Args:
            x_init (:obj:`Tensor`): The initial state of the system.
            current_x (:obj:`Tensor`, optinal): The current states of the system along a
                trajectory.
            current_u (:obj:`Tensor`, optinal): The current inputs of the system along a
                trajectory.
            time (:obj:`Tensor`, optinal): The reference time step of the dynamical system.

        Returns:
            List of :obj:`Tensor`: A list of tensors including the solved state sequence
            :math:`\mathbf{x}`, the solved input sequence :math:`\mathbf{u}`, and the associated
            quadratic costs :math:`\mathbf{c}` over the time horizon.
        '''
        K, k = self.lqr_backward(current_x, current_u, time)
        x, u, cost = self.lqr_forward(x_init, current_x, current_u, K, k)
        return x, u, cost

    def lqr_backward(self, current_x, current_u, time):

        # Q: (n_batch*, T, N, N), p: (n_batch*, T, N), where n_batch* can be any batch dimensions, e.g., (2, 3)
        n_batch = self.p.shape[:-2]
        if current_x is not None:
            ns, nc = current_x.size(-1), current_u.size(-1)
        else:
            ns, nc = self.system.B.size(-2), self.system.B.size(-1)

        K = torch.zeros(n_batch + (self.T, nc, ns), dtype=self.p.dtype, device=self.p.device)
        k = torch.zeros(n_batch + (self.T, nc), dtype=self.p.dtype, device=self.p.device)

        for t in range(self.T-1, -1, -1):
            if t == self.T - 1:
                Qt = self.Q[...,t,:,:]
                qt = self.p[...,t,:]
            else:
                if current_x is not None:
                    self.system.set_refpoint(state=current_x[...,t,:], input=current_u[...,t,:], t=time[t])
                    A = self.system.A.squeeze(-2)
                    B = self.system.B.squeeze(-1)
                    F = torch.cat((A, B), dim=-1)
                    c1 = None
                else:
                    self.system.set_refpoint(t=t)
                    F = torch.cat((self.system.A, self.system.B), dim=-1)
                    c1 = self.system.c1
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
            v = qx  + bmv(Qxu, kt) + bmv(Kt.mT, qu) + bmv(Kt.mT @ Quu, kt)

        return K, k

    def lqr_forward(self, x_init, current_x, current_u, K, k):

        assert x_init.device == K.device == k.device
        assert x_init.dtype == K.dtype == k.dtype
        assert x_init.ndim == 2, "Shape not compatible."

        # Q: (n_batch*, T, N, N), p: (n_batch*, T, N), where n_batch* can be any batch dimensions, e.g., (2, 3)
        n_batch = self.p.shape[:-2]
        if current_x is not None:
            ns, nc = current_x.size(-1), current_u.size(-1)
        else:
            ns, nc = self.system.B.size(-2), self.system.B.size(-1)

        u = torch.zeros(n_batch + (self.T, nc), dtype=self.p.dtype, device=self.p.device)
        delta_u = torch.zeros(n_batch + (self.T, nc), dtype=self.p.dtype, device=self.p.device)
        cost = torch.zeros(n_batch, dtype=self.p.dtype, device=self.p.device)
        delta_xt = torch.zeros_like(x_init, dtype=x_init.dtype, device=self.p.device)
        x = torch.zeros(n_batch + (self.T+1, ns), dtype=self.p.dtype, device=self.p.device)
        x[..., 0, :] = x_init
        xt = x_init

        for t in range(self.T):
            Kt, kt = K[...,t,:,:], k[...,t,:]
            delta_u[..., t, :] = delta_ut = bmv(Kt, delta_xt) + kt
            if current_x is None:
                u[...,t,:] = ut = delta_ut + bmv(Kt, xt)
            else:
                u[...,t,:] = ut = delta_ut + current_u[...,t,:]
            xut = torch.cat((xt, ut), dim=-1)
            x[...,t+1,:] = xt = self.system(xt, ut)[0]
            cost = cost + 0.5 * bvmv(xut, self.Q[...,t,:,:], xut) + (xut * self.p[...,t,:]).sum(-1)

        return x[...,0:-1,:], u, cost
