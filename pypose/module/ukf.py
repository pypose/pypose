import torch
from .. import bmv
from .ekf import EKF
from torch.linalg import pinv


class UKF(EKF):
    r'''
    Performs Batched Unscented Kalman Filter (UKF).

    Args:
        model (:obj:`System`): The system model to be estimated, a subclass of
            :obj:`pypose.module.NLS`.
        Q (:obj:`Tensor`, optional): The covariance matrices of system transition noise.
            Ignored if provided during each iteration. Default: ``None``
        R (:obj:`Tensor`, optional): The covariance matrices of system observation noise.
            Ignored if provided during each iteration. Default: ``None``
        msqrt (callable, optional): Matrix square-root function returning a factor
            ``L`` such that ``L @ L.mT`` equals its input. Default: ``torch.linalg.cholesky``.

    Batch dimensions precede vector and matrix dimensions. For batch shape ``B``,
    states have shape ``(*B, n)``, observations ``(*B, m)``, and inputs
    ``(*B, l)``. Covariances ``P`` and ``Q`` have shape ``(*B, n, n)``,
    and ``R`` has shape ``(*B, m, m)``. Covariances can be shared across
    batches using shapes ``(n, n)`` and ``(m, m)``. An empty ``B`` gives an
    unbatched filter. Batch entries represent independent systems, not time steps.

    Internally, the sigma-point dimension comes first. Model methods receive
    states of shape ``(S, *B, n)`` and return states or observations of shape
    ``(S, *B, n)`` or ``(S, *B, m)``. Inputs retain shape ``(*B, l)`` and
    broadcast across samples. Model operations must preserve batch independence.
    Here ``S = 2*n+1``. Returned estimates have no sample dimension.

    A non-linear system can be described as

    .. math::
        \begin{aligned}
            \mathbf{x}_{k+1} &= \mathbf{f}(\mathbf{x}_k, \mathbf{u}_k, t_k) + \mathbf{w}_k,
            \quad \mathbf{w}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{Q})  \\
            \mathbf{y}_{k+1} &= \mathbf{g}(\mathbf{x}_{k+1}, \mathbf{u}_k, t_k) + \mathbf{v}_{k+1},
            \quad \mathbf{v}_{k+1} \sim \mathcal{N}(\mathbf{0}, \mathbf{R})
        \end{aligned}

    UKF can be described as the following equations. The sigma-point index :math:`i`
    identifies a sample, and the time indices indicate the state or measurement step.

    1. Sigma Points.

      .. math::
          \begin{aligned}
              & \mathbf{x}_k^{\left ( i \right ) } = \mathbf{x}_k^{+} +
                  \mathbf{\check{x}}_k^{\left(i \right)}, & \quad i=0,...,2n\\
              & \mathbf{\check{x}}_k^{\left (i \right)} = \mathbf{0}, & \quad i = 0 \\
              & \mathbf{\check{x}}_k^{\left (i \right)} =
                  \left(\sqrt{(n+\kappa)P_k} \right)_{i}^{T},
               & \quad i= 1, ..., n \\
              & \mathbf{\check{x}}_k^{\left(i \right)} =
                  -\left(\sqrt{(n+\kappa)P_k} \right)_{i}^{T},
                   & \quad i=n+1, ..., 2n \\
         \end{aligned}

    where :math:`\left(\sqrt{(n+\kappa)P_k}\right) _{i}` is the :math:`i`-th row of
    :math:`\left(\sqrt{(n+\kappa)P_k}\right)` and :math:`n` is the dimension of state :math:`\mathbf{x}`.
    Here the square-root matrix is :math:`L^T`, where
    :math:`LL^T=(n+\kappa)P_k`, so its rows are the sigma-point offsets.

    Their weighting cofficients are given as

      .. math::
          \begin{aligned}
              &W^{(0)} = \frac{\kappa}{n+\kappa}, & \quad i = 0\\
              &W^{(i)} = \frac{1}{2(n+\kappa)}, & \quad i = 1,...,2n\\
          \end{aligned}

    2. Priori State Estimation.

      .. math::
           \begin{aligned}
               \mathbf{x}_{k+1}^{(i)} &= f(\mathbf{x}_k^{(i)}, \mathbf{u}_k, t_k),
                   & \quad i = 0, \cdots, 2n \\
               \mathbf{x}_{k+1}^{-} &= \sum_{i=0}^{2n} W^{(i)} \mathbf{x}_{k+1}^{(i)}
           \end{aligned}

    3. Priori Covariance.

      .. math::
          \mathbf{P}_{k+1}^{-} = \sum_{i=0}^{2n} W^{(i)}
              \left(\mathbf{x}_{k+1}^{(i)} - \mathbf{x}_{k+1}^{-} \right)
              \left(\mathbf{x}_{k+1}^{(i)} - \mathbf{x}_{k+1}^- \right)^T + \mathbf{Q}

    4. Observational Estimation.

      Regenerate sigma points :math:`\tilde{\mathbf{x}}_{k+1}^{(i)}` from
      :math:`(\mathbf{x}_{k+1}^{-}, \mathbf{P}_{k+1}^{-})` using step 1.
      These points include the process-noise covariance added in step 3.

      .. math::
          \begin{aligned}
              & \mathbf{y}_{k+1}^{(i)} = g \left( \tilde{\mathbf{x}}_{k+1}^{(i)}, \mathbf{u}_k, t_k \right),
                  & \quad i = 0, \cdots, 2n \\
              & \bar{\mathbf{y}}_{k+1} = \sum_{i=0}^{2n} W^{(i)} \mathbf{y}_{k+1}^{(i)}
          \end{aligned}

    5. Observational Covariance.

      .. math::
          \mathbf{P}_{y_{k+1}} = \sum_{i=0}^{2n} W^{(i)}
              \left(\mathbf{y}_{k+1}^{(i)} - \bar{\mathbf{y}}_{k+1} \right)
              \left( \mathbf{y}_{k+1}^{(i)} - \bar{\mathbf{y}}_{k+1} \right)^T + \mathbf{R}

    6. Priori and Observation Covariance:

      .. math::
          \mathbf{P}_{x_{k+1}y_{k+1}} = \sum_{i=0}^{2n} W^{(i)}
              \left( \tilde{\mathbf{x}}_{k+1}^{(i)} - \mathbf{x}_{k+1}^- \right)
              \left( \mathbf{y}_{k+1}^{(i)} - \bar{\mathbf{y}}_{k+1} \right)^T

    7. Kalman Gain

      .. math::
          \mathbf{K}_{k+1} = \mathbf{P}_{x_{k+1}y_{k+1}}\mathbf{P}_{y_{k+1}}^{-1}

    8. Posteriori State Estimation.

      .. math::
          \mathbf{x}_{k+1}^{+} = \mathbf{x}_{k+1}^{-} +
              \mathbf{K}_{k+1} (\mathbf{y}_{k+1}- \bar{\mathbf{y}}_{k+1})

    9. Posteriori Covariance Estimation.

      .. math::
          \mathbf{P}_{k+1} = \mathbf{P}_{k+1}^{-} -
              \mathbf{K}_{k+1}\mathbf{P}_{y_{k+1}}\mathbf{K}_{k+1}^T

    where superscript :math:`\cdot^{-}` and :math:`\cdot^{+}` denote the priori and
    posteriori estimation, respectively.

    Example:
        1. Define a discrete-time non-linear system (NLS) model

        >>> import torch, pypose as pp
        >>> class NLS(pp.module.NLS):
        ...     def __init__(self):
        ...         super().__init__()
        ...
        ...     def forward(self, state, input, noise):
        ...         state = self.state_transition(state, input, self.systime) + noise
        ...         return state, self.observation(state, input, self.systime + 1)
        ...
        ...     def state_transition(self, state, input, t=None):
        ...         return state.cos() + input
        ...
        ...     def observation(self, state, input, t=None):
        ...         return state.sin() + input

        2. Create a model and filter

        >>> model = NLS()
        >>> ukf = pp.module.UKF(model)

        3. Prepare data

        >>> B, T, N = 3, 5, 2 # batchsize, steps, state dim
        >>> states = torch.zeros(B, T, N)
        >>> inputs = torch.randn(B, T, N)
        >>> observ = torch.zeros(B, T, N)
        >>> # std of transition, observation, and estimation
        >>> q, r, p = 0.1, 0.1, 10
        >>> Q = torch.eye(N).repeat(B, 1, 1) * q**2
        >>> R = torch.eye(N).repeat(B, 1, 1) * r**2
        >>> P = torch.eye(N).repeat(B, T, 1, 1) * p**2
        >>> estim = torch.randn(B, T, N) * p

        4. Predict the next state and correct it using a measurement of that same state.

        >>> for i in range(T - 1):
        ...     w = q * torch.randn(B, N) # transition noise
        ...     v = r * torch.randn(B, N) # observation noise
        ...     states[:, i+1], observ[:, i+1] = model(states[:, i], inputs[:, i], w)
        ...     estim[:, i+1], P[:, i+1] = ukf(estim[:, i], observ[:, i+1] + v, inputs[:, i], P[:, i], Q, R)
        ... print('Est error:', (states - estim).norm(dim=-1))
        Est error: tensor(...)

    Note:
        Implementation is based on Section 14.3 of this book

        * Dan Simon, `Optimal State Estimation: Kalman, H∞, and Nonlinear Approaches
          <https://onlinelibrary.wiley.com/doi/book/10.1002/0470045345>`_,
          Cleveland State University, 2006
    '''

    def __init__(self, model, Q=None, R=None, msqrt=None):
        super().__init__(model, Q, R)
        self.msqrt = torch.linalg.cholesky if msqrt is None else msqrt

    def forward(self, x, y, u, P, Q=None, R=None, t=None, k=None):
        r'''
        Performs one step estimation.

        The measurement ``y`` must correspond to the state after applying the
        transition to ``x`` with input ``u``.

        Args:
            x (:obj:`Tensor`): estimated system state of previous step, shape ``(*B, n)``.
            y (:obj:`Tensor`): system observation at current step, shape ``(*B, m)``.
            u (:obj:`Tensor`): system input at current step, shape ``(*B, l)``.
            P (:obj:`Tensor`): state estimation covariance of previous step, shape
                ``(*B, n, n)`` or ``(n, n)``.
            Q (:obj:`Tensor`, optional): covariance of system transition model.
            R (:obj:`Tensor`, optional): covariance of system observation model.
            t (:obj:`Tensor`, optional): timestamp for ths system to estimate. Default: ``None``.
            k (``int``, optional): a parameter for weighting the sigma points.
                If ``None`` is given, then ``3 - n`` is adopted (``n`` is the state dimension),
                which is best for Gaussian noise model. Default: ``None``.

        Return:
            tuple of :obj:`Tensor`: posteriori state ``(*B, n)`` and covariance
            ``(*B, n, n)``, where ``B`` is the broadcast batch shape.
        '''
        # Upper cases are matrices, lower cases are vectors
        k = 3 - x.size(-1) if k is None else k
        Q = Q if Q is not None else self.Q
        R = R if R is not None else self.R
        self.model.set_refpoint(state=x, input=u, t=t)

        xs, w = self.sigma_weight_points(x, P, k)
        xs = self.model.state_transition(xs, u, t)
        xe = (w * xs).sum(dim=0)
        ex = xe - xs
        P = self.compute_cov(ex, ex, w, Q)

        xs, w = self.sigma_weight_points(xe, P, k)
        ex = xe - xs
        ys = self.model.observation(xs, u, t)
        ye = (w * ys).sum(dim=0)
        ey = ye - ys
        Py = self.compute_cov(ey, ey, w, R)

        Pxy = self.compute_cov(ex, ey, w)
        K = Pxy @ pinv(Py)
        x = xe + bmv(K, y - ye)
        P = P - K @ Py @ K.mT

        return x, P

    def sigma_weight_points(self, x, P, k):
        r'''
        Compute sigma point and its weights

        Args:
            x (:obj:`Tensor`): estimated system state of previous step
            P (:obj:`Tensor`): state estimation covariance of previous step
            k (:obj:`int`): parameter for weighting sigma points.

        Return:
            tuple of :obj:`Tensor`: sigma points ``(2*n+1, *B, n)`` and
            weights ``(2*n+1, *B, 1)``.
        '''
        assert x.size(-1) == P.size(-1) == P.size(-2), 'Invalid shape'
        batch = torch.broadcast_shapes(x.shape[:-1], P.shape[:-2])
        x = x.expand(batch + x.shape[-1:])
        P = P.expand(batch + P.shape[-2:])
        n, xe = x.size(-1), x.unsqueeze(0)
        # Factor columns become sigma-point offsets on the leading sample axis.
        xr = self.msqrt((n + k) * P).mT.movedim(-2, 0)
        we = torch.full(xe.shape[:-1], k / (n + k), dtype=x.dtype, device=x.device)
        wr = torch.full(xr.shape[:-1], 1 / (2 * (n + k)), dtype=x.dtype, device=x.device)
        p = torch.cat((xe, xe + xr, xe - xr), dim=0)
        w = torch.cat((we, wr, wr), dim=0)
        return p, w.unsqueeze(-1)

    def compute_cov(self, a, b, w, Q=0):
        '''Compute covariance of two set of variables.'''
        a, b = a.unsqueeze(-1), b.unsqueeze(-1)
        return Q + (w.unsqueeze(-1) * a @ b.mT).sum(dim=0)
