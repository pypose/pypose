import torch
from torch import nn
from ..basics import bmv, MatrixSquareRoot
from torch.linalg import pinv


class UKF(nn.Module):
    r'''
    Performs Batched Unscented Kalman Filter (UKF).

    Args:
        model (:obj:`System`): The system model to be estimated, a subclass of
            :obj:`pypose.module.System`.
        Q (:obj:`Tensor`, optional): The covariance matrices of system transition noise.
            Ignored if provided during each iteration. Default: ``None``
        R (:obj:`Tensor`, optional): The covariance matrices of system observation noise.
            Ignored if provided during each iteration. Default: ``None``
        matrix_square_root_device(:obj:`String`, optional): The compute type of  Differentiable Matrix Square Root. Default: ``cpu``


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
            \mathbf{z}_{k+1} &= \mathbf{A}_{k}\mathbf{x}_{k} + \mathbf{B}_{k}\mathbf{u}_{k}
                             + \mathbf{c}_{k}^1 + \mathbf{w}_k\\
            \mathbf{y}_{k} &= \mathbf{C}_{k}\mathbf{x}_{k} + \mathbf{D}_{k}\mathbf{u}_{k}
                           + \mathbf{c}_{k}^2 + \mathbf{v}_k\\
        \end{align*}

    UKF can be described as the following five equations, where the subscript :math:`\cdot_{k}`
    is omited for simplicity.

    1.Sigma Point.

        .. math::
            \begin{align*}
                \mathbf{x}^{\left ( i \right ) } = \mathbf{x}^{+} +  \mathbf{\check{x }}^{\left ( i \right ) },
                \quad i=\quad1,...,2n\\
                \mathbf{\check{x}}^{\left ( i \right ) }  = \left ( \sqrt{nP}  \right ) _{i}^{T},
                 \quad i=\quad1,...,n\\
                \mathbf{\check{x}}^{\left ( n+i \right ) }  = -\left ( \sqrt{nP}  \right ) _{i}^{T},
                 \quad i=\quad1,...,n\\
            \end{align*}

    2. Priori State Estimation.

        .. math::
             \mathbf{x}^{-} = \frac{1}{2n} \sum_{i=1}^{2n} \left ( f\left ( x^{i},u,t  \right ) +w_{k}  \right )

    3. Priori Covariance.

        .. math::
            \mathbf{P}_{k}^{-} = \frac{1}{2} \sum_{1}^{2n}\left ( x^{\left ( i \right ) }-x^{-}  \right )  \left ( x^{\left ( i \right )} -x^- \right )^T+\mathbf{Q}_{k-1}

    4.Observational estimation

        .. math::
            \check{y} = \frac{1}{2n} \sum_{i=1}^{2n}\left (g\left ( x^{i},u,t \right ) +\mathbf{R}_{k}   \right  )
    5 Observational Covariance.

        .. math::
            \mathbf{P}_{y} = \frac{1}{2n} \sum_{i=1}^{2n}\left ( y^{\left ( i \right ) }- \check{y}  \right )  \left ( y^{\left ( i \right )} - \check{y} \right )^T+\mathbf{R}_{k}
    6 Priori and Observation Covariance :

        .. math::
            \mathbf{P}_{xy} = \frac{1}{2n} \sum_{i=1}^{2n}\left ( x^{\left ( i \right ) }-x^-  \right )  \left ( y^{\left ( i \right )} -\check{y} \right )^T
    3. Update Kalman Gain

        .. math::
            \mathbf{K} = \mathbf{P}_{xy}\mathbf{P}_{y}^{-1}

    4. Posteriori State Estimation

        .. math::
            \mathbf{x}^{+} = \mathbf{x}^{-} + \mathbf{K} (y-\check{y} )

    5. Posteriori Covariance Estimation

        .. math::
            \mathbf{P} = \mathbf{P}_{k}^{-} - \mathbf{K}\mathbf{P}_{y}\mathbf{K}_T

    where superscript :math:`\cdot^{-}` and :math:`\cdot^{+}` denote the priori and
    posteriori estimation, respectively.

    Example:
        1. Define a Nonlinear Time Invariant (NTI) system model

        >>> import torch, pypose as pp
        >>> class NTI(pp.module.System):
        ...     def __init__(self):
        ...         super().__init__()
        ...
        ...     def state_transition(self, state, input, t=None):
        ...         return state.cos() + input
        ...
        ...     def observation(self, state, input, t):
        ...         return state.sin() + input

        2. Create a model and filter

        >>> model = NTI()
        >>> ukf = pp.module.UKF(model)

        3. Prepare data

        >>> T, N = 5, 2 # steps, state dim
        >>> states = torch.zeros(T, N)
        >>> inputs = torch.randn(T, N)
        >>> observ = torch.zeros(T, N)
        >>> # std of transition, observation, and estimation
        >>> q, r, p = 0.1, 0.1, 10
        >>> Q = torch.eye(N) * q**2
        >>> R = torch.eye(N) * r**2
        >>> P = torch.eye(N).repeat(T, 1, 1) * p**2
        >>> estim = torch.randn(T, N) * p

        4. Perform UKF prediction. Note that estimation error becomes smaller with more steps.

        >>> for i in range(T - 1):
        ...     w = q * torch.randn(N) # transition noise
        ...     v = r * torch.randn(N) # observation noise
        ...     states[i+1], observ[i] = model(states[i] + w, inputs[i])
        ...     estim[i+1], P[i+1] = ukf(estim[i], observ[i] + v, inputs[i], P[i], Q, R)
        ... print('Est error:', (states - estim).norm(dim=-1))
        Est error: tensor([8.8161, 9.0322, 5.4756, 2.2453, 0.9141])

    Warning:
        Don't introduce noise in ``System`` methods ``state_transition`` and ``observation``
        for filter testing, as those methods are used for automatically linearizing the system
        by the parent class ``pypose.module.System``, unless your system model explicitly
        introduces noise.

    Note:
        Implementation is based on Section 14.3 of this book

        * Dan Simon, `Optimal State Estimation: Kalman, H∞, and Nonlinear Approaches
          <https://onlinelibrary.wiley.com/doi/epdf/10.1002/0470045345.fmatter>`_,
          Cleveland State University, 2006
    '''

    def __init__(self, model, Q=None, R=None):
        super().__init__()
        self.set_uncertainty(Q=Q, R=R)
        self.model = model

    def compute_weight(self):
        r'''
        compute ukf weight

        Return:
        out (:obj:`int`): the ukf weight.
        '''
        return 1 / (2 * self.dim)

    def compute_sigma(self, x, u, P, C, D, c2):  # compute sigma point
        r'''
        compute sigma point and Observation

        Args:
            x (:obj:`Tensor`): estimated system state of previous step
            u (:obj:`Tensor`): system input at current step
            P (:obj:`Tensor`): state estimation covariance of previous step
            C (:obj:`Tensor`): The output matrix of LTI system.
            D (:obj:`Tensor`): The observation matrix of LTI system,
            c2 (:obj:`Tensor`): The constant output of LTI system.

        Return:
            list of :obj:`Tensor`: sigma point and Oerservation
        '''
        sqrtm = MatrixSquareRoot()
        # gather index
        index_repeat = tuple(
            torch.cat([torch.tensor([2], dtype=torch.int64), torch.ones(self.dim - 1, dtype=torch.int64)]).numpy())
        index_weight = torch.arange(self.dim, device=P.device, dtype=P.dtype).unsqueeze(1)
        index = torch.ones(self.dim, self.dim, device=P.device, dtype=P.dtype)

        index_gather = torch.as_tensor(index * index_weight, dtype=torch.int64, device=P.device).repeat(
            index_repeat).unsqueeze(1).reshape(2 * self.dim, 1, self.dim)

        # compute root of P
        repeat_dim = tuple(torch.cat([torch.tensor([self.dim * 2]), torch.tensor(P.shape)]).numpy())  # repeat dim
        p_expand = P.expand(repeat_dim)

        np_expand = self.dim * p_expand  # calculate np
        np_expand_norm = torch.linalg.norm(np_expand, dim=[1, 2]).reshape(np_expand.size(0), 1, 1)
        I_np = torch.eye(np_expand.size(1), requires_grad=False, device=np_expand.device).reshape(1, np_expand.size(1),
                                                                                                  np_expand.size(
                                                                                                      1)).repeat(
            np_expand.size(0), 1, 1)

        np_expand_sqrt = sqrtm.matrix_pade_approximant(np_expand, np_expand_norm, I_np)  # square root of np
        np_expand_sqrt[self.dim:] *= -1
        np_select = np_expand_sqrt.gather(1, index_gather)  # select point from np_expand_sqrt

        # compute sigma point
        x_sigma = x + np_select
        y_sigma = bmv(C, x_sigma) + bmv(D, u) + c2

        return x_sigma.reshape(-1, self.dim), y_sigma.reshape(-1, self.dim)

    def compute_conv_mix(self, x_estimate, x_sigma, y_estimate, y_sigma):
        r'''
        compute mix covariance

        Args:
            x_estimate (:obj:`Tensor`):  System prior state estimation.
            x_sigma (:obj:`Tensor`):  Sigma point of estimation.
            y_estimate (:obj:`Tensor`): Observation  estimation.
            y_sigma (:obj:`Tensor`): Observation estimation of sigma point .

        Return:
            out (:obj:`Tensor`): Mix covariance pf prior state estimation   .
        '''

        e_x = torch.sub(x_sigma, x_estimate).unsqueeze(2)
        e_y = torch.sub(y_sigma, y_estimate).unsqueeze(2)
        p_estimate = torch.sum(torch.bmm(e_x, e_y.permute(0, 2, 1)), dim=0)

        return self.weight * p_estimate

    def compute_conv(self, estimate, sigma, noise=0):
        r'''
        compute covariance

        Args:
            estimate (:obj:`Tensor`): System estimate.
            sigma (:obj:`Tensor`):  Sigma point of estimate.

        Return:
            out (:obj:`Tensor`): the covariance of  prior state estimation .
        '''

        e = torch.sub(sigma, estimate).unsqueeze(2)
        p_estimate = torch.sum(torch.bmm(e, e.permute(0, 2, 1)), dim=0)

        return self.weight * p_estimate + noise

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
        A, B = self.model.A, self.model.B
        C, D = self.model.C, self.model.D
        c1, c2 = self.model.c1, self.model.c2
        Q = Q if Q is not None else self.Q
        R = R if R is not None else self.R
        # x = bmv(A, x) + bmv(B, u) + c1

        self.dim = x.shape[0]
        self.weight = self.compute_weight()
        self.loop_range = self.dim * 2 + 1

        # compute sigma point,mean,covariance
        x_sigma, y_sigma = self.compute_sigma(x, u, P, C, D, c2)
        x_sigma = bmv(A, x_sigma) + bmv(B, u) + c1
        x_estimate = self.weight * torch.sum(x_sigma, dim=0)
        Pk = self.compute_conv(x_estimate, x_sigma, Q)
        x_sigma, y_sigma = self.compute_sigma(x_estimate, u, Pk, C, D, c2)

        y_estimate = self.weight * torch.sum(y_sigma, dim=0)
        Py = self.compute_conv(y_estimate, y_sigma, R)
        Pxy = self.compute_conv_mix(x_estimate, x_sigma, y_estimate, y_sigma)

        # Equation
        K = Pxy @ pinv(Py)
        e = y - y_estimate
        x = x_estimate + bmv(K, e)
        P = Pk - K @ Py @ K.mT

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
