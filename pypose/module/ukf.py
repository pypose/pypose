import torch
from torch import nn
from ..basics import bmv
from torch.linalg import pinv
from pypose.basics.utils import *


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
            \mathbf{z}_{k+1} = \mathbf{A}_{k}\mathbf{x}_{k} + \mathbf{B}_{k}\mathbf{u}_{k}
                             + \mathbf{c}_{k}^1 + \mathbf{w}_k\\
            \mathbf{y}_{k} = \mathbf{C}_{k}\mathbf{x}_{k} + \mathbf{D}_{k}\mathbf{u}_{k}
                           + \mathbf{c}_{k}^2 + \mathbf{v}_k\\
        \end{align*}

    UKF can be described as the following five equations, where the subscript :math:`\cdot_{k}`
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
        Est error: tensor([10.9069,  9.6868,  1.7228,  1.6782,  0.5352])

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

    def __init__(self, model, Q=None, R=None, matrix_square_root_device='cpu'):
        super().__init__()
        self.set_uncertainty(Q=Q, R=R)
        self.model = model
        self.matrix_square_root_device = matrix_square_root_device

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
        matrix_sqrt = MPA_Lya.apply

        # gather index
        index_repeat =  tuple(torch.cat([torch.tensor([2],dtype=torch.int64), torch.ones(self.dim-1,dtype=torch.int64)]).numpy())
        index_weight = torch.arange(self.dim,device=P.device,dtype=P.dtype).unsqueeze(1)
        index = torch.ones(self.dim,self.dim,device=P.device,dtype=P.dtype)
        index_finall = torch.tensor(index * index_weight,dtype=torch.int64,device=P.device).repeat(index_repeat).unsqueeze(1).reshape(2*self.dim,1,self.dim)


        repeat_dim = tuple(torch.cat([torch.tensor([self.dim * 2]), torch.tensor(P.shape)]).numpy()) #repeat dim
        p_repeat = P.expand(repeat_dim)
        np_repeat = self.dim * p_repeat  # np calculate
        np_repeat = matrix_sqrt([np_repeat, self.matrix_square_root_device]) #square root of np
        np_repeat[self.dim:] *= -1
        np_repeat_select = np_repeat.gather(1, index_finall) #select point from np_repeat


        x_sigma1 = x+np_repeat_select
        y_sigma1 = bmv(C,x_sigma1) + bmv(D,u) + c2

        x_sigma = []
        y_sigma = []
        for loop in range(1, self.loop_range):


            nP = (self.dim * P).unsqueeze(0)
            param = nP, self.matrix_square_root_device

            if loop <= self.dim:

                x_ = x + matrix_sqrt(param)[0].mT[loop - 1]
            else:

                x_ = x - matrix_sqrt(param)[0].mT[loop - self.dim - 1]

            y_ = bmv(C, x_) + bmv(D, u) + c2  # compute Observation
            x_sigma.append(x_)
            y_sigma.append(y_)
        x_sigma = torch.cat(x_sigma, dim=0).reshape(-1, self.dim)
        y_sigma = torch.cat(y_sigma, dim=0).reshape(-1, self.dim)
        print(sum(x_sigma.view(-1)-x_sigma1.view(-1)),sum(y_sigma.view(-1)-y_sigma1.view(-1)))

        return x_sigma1.reshape(-1, self.dim), y_sigma1.reshape(-1, self.dim)

    def compute_conv_mix(self, x_estimate, x_sigma, y_estimate, y_sigma):
        r'''
        compute mix covariance

        Args:
        x_estimate (:obj:`Tensor`): matrices to be multiplied.
        x_sigma (:obj:`Tensor`): vectors to be multiplied.
        y_estimate (:obj:`Tensor`): matrices to be multiplied.
        y_sigma (:obj:`Tensor`): vectors to be multiplied.

        Return:
            out (:obj:`Tensor`): the mix covariance of x,y .
        '''

        e_x = torch.sub(x_sigma, x_estimate).unsqueeze(2)
        e_y = torch.sub(y_sigma, y_estimate).unsqueeze(2)
        p_estimate = torch.sum(torch.bmm(e_x, e_y.permute(0, 2, 1)), dim=0)

        return self.weight * p_estimate

    def compute_conv(self, estimate, sigma, noise=0):
        r'''
        compute covariance

        Args:
        estimate (:obj:`Tensor`): estimate.
        sigma (:obj:`Tensor`):  sigma point of  estimate.

        Return:
            out (:obj:`Tensor`): the  covariance of estimate .
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
        x = bmv(A, x) + bmv(B, u) + c1

        self.dim = x.shape[0]
        self.weight = self.compute_weight()
        self.loop_range = self.dim * 2 + 1

        # compute sigma point,mean,covariance
        x_sigma, y_sigma = self.compute_sigma(x, u, P, C, D, c2)
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
