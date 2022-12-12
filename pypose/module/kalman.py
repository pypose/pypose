from torch import nn
import torch, pypose as pp
from torch.linalg import pinv


class EKF(nn.Module):
    r'''
    Performs Batched Extended Kalman Filter (EKF).

    Args:
        model (:obj:`System`): The system model to be estimated, a subclass of
            :obj:`pypose.module.System`.
        Q (:obj:`Tensor`, optional): The covariance matrices of system transition noise.
            Ignored if provided during each iteration. Default: ``None``
        R (:obj:`Tensor`, optional): The covariance matrices of system observation noise.
            Ignored if provided during each iteration. Default: ``None``

    A non-linear system can be described as

    .. math::
        \begin{aligned}
            \mathbf{x}_{k+1} &= \mathbf{f}(\mathbf{x}_k, \mathbf{u}_k, t_k) + \mathbf{w}_k,
            \quad \mathbf{w}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{Q})  \\
            \mathbf{y}_{k} &= \mathbf{g}(\mathbf{x}_k, \mathbf{u}_k, t_k) + \mathbf{v}_k,
            \quad \mathbf{v}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{R})
        \end{aligned}

    It will be linearized automatically:

    .. math::
        \begin{align*}
            \mathbf{z}_{k+1} = \mathbf{A}_{k}\mathbf{x}_{k} + \mathbf{B}_{k}\mathbf{u}_{k}
                             + \mathbf{c}_{k}^1 + \mathbf{w}_k\\
            \mathbf{y}_{k} = \mathbf{C}_{k}\mathbf{x}_{k} + \mathbf{D}_{k}\mathbf{u}_{k}
                           + \mathbf{c}_{k}^2 + \mathbf{v}_k\\
        \end{align*}

    EKF can be described as the following five equations, where the subscript :math:`\cdot_{k}`
    is omited for simplicity.

    1. Priori State Estimation.

        .. math::
            \mathbf{x}^{-} = \mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{u}_k + \mathbf{c}_1

    2. Priori Covariance Propagation.

        .. math::
            \mathbf{P}^{-} = \mathbf{A}\mathbf{P}\mathbf{A}^{T} + \mathbf{Q}

    3. Update Kalman Gain

        .. math::
            \mathbf{K} = \mathbf{P}\mathbf{C}^{T}
                        (\mathbf{C}\mathbf{P} \mathbf{C}^{T} + \mathbf{R})^{-1}

    4. Posteriori State Estimation

        .. math::
            \mathbf{x}^{+} = \mathbf{x}^{-} + \mathbf{K} (\mathbf{y} -
                        \mathbf{C}\mathbf{x}^{-} - \mathbf{D}\mathbf{u} - \mathbf{c}_2)

    5. Posteriori Covariance Estimation

        .. math::
            \mathbf{P} = (\mathbf{I} - \mathbf{K}\mathbf{C}) \mathbf{P}^{-}

    where superscript :math:`\cdot^{-}` and :math:`\cdot^{+}` denote the priori and
    posteriori estimation, respectively.

    Example:
        1. Initialize model and filter

        >>> import torch, pypose as pp
        >>> model = pp.module.System()  # System model
        >>> filter = pp.module.EKF(model)

        2. Calculate filter for one iteration

        >>> # u is system input
        >>> state, observation = model(state, u)  # model measurement
        >>> est_state, P = filter(est_state, observation, P, u)

    Note:
        Implementation is based on Section 5.1 of this book

        * Dan Simon, `Optimal State Estimation: Kalman, H∞, and Nonlinear Approaches
          <https://onlinelibrary.wiley.com/doi/epdf/10.1002/0470045345.fmatter>`_,
          Cleveland State University, 2006
    '''
    def __init__(self, model, Q=None, R=None):
        super().__init__()
        self.set_uncertainty(Q=Q, R=R)
        self.model = model

    def forward(self, x, y, u, P, Q=None, R=None):
        r'''
        Performs one step estimation.

        Args:
            x (:obj:`Tensor`): estimated system state of previous step
            y (:obj:`Tensor`): system observation at current step (measurement)
            u (:obj:`Tensor`): system input at current step
            P (:obj:`Tensor`): state estimation covariance of previous step
            Q (:obj:`Tensor`, optional): covariance of system transition model
            R (:obj:`Tensor`, optional): covariance of system observation model

        Return:
            list of :obj:`Tensor`: posteriori state and covariance estimation
        '''
        # Upper cases are matrices, lower cases are vectors
        self.model.set_refpoint(state=x, input=u)
        I = torch.eye(P.shape[-1], device=P.device, dtype=P.dtype)
        A, B = self.model.A, self.model.B
        C, D = self.model.C, self.model.D
        c1, c2 = self.model.c1, self.model.c2
        Q = Q if Q is not None else self.Q
        R = R if R is not None else self.R

        x = pp.bmv(A, x) + pp.bmv(B, u) + c1      # 1. System transition
        P = A @ P @ A.mT + Q                      # 2. Covariance predict
        K = P @ C.mT @ pinv(C @ P @ C.mT + R)     # 3. Kalman gain
        e = y - pp.bmv(C, x) - pp.bmv(D, u) - c2  #    predicted observation error
        x = x + pp.bmv(K, e)                      # 4. Posteriori state
        P = (I - K @ C) @ P                       # 5. Posteriori covariance
        return x, P

    @property
    def Q(self):
        r'''
        The covariance of system transition noise.
        '''
        if not hasattr(self, '_Q'):
            raise NotImplementedError('Call set_uncertainty() to define\
                                        transition covariance Q.')
        return self._Q

    @property
    def R(self):
        r'''
        The covariance of system observation noise.
        '''
        if not hasattr(self, '_R'):
            raise NotImplementedError('Call set_uncertainty() to define\
                                        transition covariance R.')
        return self._R

    def set_uncertainty(self, Q=None, R=None):
        r'''
        Set the covariance matrices of transition noise and observation noise.

        Args:
            Q (:obj:`Tensor`): batched square covariance matrices of transition noise.
            R (:obj:`Tensor`): batched square covariance matrices of observation noise.
        '''
        if Q is not None:
            self.register_buffer("_Q", Q)
        if R is not None:
            self.register_buffer("_R", R)
