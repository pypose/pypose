import torch
from .. import bmv
from torch import nn
from torch.autograd.functional import jacobian


class System(nn.Module):
    r'''
    The base class for a system dynamics model.

    In most of the cases, users only need to subclass a specific dynamic system,
    such as linear time invariant system :meth:`LTI`, Linear Time-Variant :meth:`LTV`,
    and a non-linear system :meth:`NLS`.
    '''
    def __init__(self):
        super().__init__()
        self.register_buffer('_t',torch.tensor(0, dtype=torch.int64))
        self.register_forward_hook(self.forward_hook)

    def forward_hook(self, module, inputs, outputs):
        r'''
        Automatically advances the time step.
        '''
        self._t.add_(1)

    def reset(self, t=0):
        self._t.fill_(t)
        return self

    def forward(self, state, input):
        r'''
        Defines the computation performed at every call that advances the system by one time step.

        Note:
            The :obj:`forward` method implicitly increments the time step via :obj:`forward_hook`.
            :obj:`state_transition` and :obj:`observation` still accept time for the flexiblity
            such as time-varying system. One can directly access the current system time via the
            property :obj:`systime` or :obj:`_t`.

        Note:
            To introduce noise in a model, redefine this method via
            subclassing. See example in ``examples/module/ekf/tank_robot.py``.
        '''
        self.state, self.input = torch.atleast_1d(state), torch.atleast_1d(input)
        state = self.state_transition(self.state, self.input)
        obs = self.observation(self.state, self.input)
        return state, obs

    def state_transition(self, state, input, t=None):
        r'''
        Args:
            state (:obj:`Tensor`): The state of the dynamical system
            input (:obj:`Tensor`): The input to the dynamical system
            t (:obj:`Tensor`): The time step of the dynamical system.  Default: ``None``.

        Returns:
            Tensor: The state of the system at next time step

        Note:
            The users need to define this method and can access the current time via the property
            :obj:`systime`. Don't introduce system transision noise in this function, as it will
            be used for linearizing the system automaticalluy.
        '''
        raise NotImplementedError("The users need to define their own state transition method")

    def observation(self, state, input, t=None):
        r'''
        Args:
            state (:obj:`Tensor`): The state of the dynamical system
            input (:obj:`Tensor`): The input to the dynamical system
            t (:obj:`Tensor`): The time step of the dynamical system.  Default: ``None``.

        Returns:
            Tensor: The observation of the system at the current step

        Note:
            The users need to define this method and can access the current system time via the
            property :obj:`systime`. Don't introduce system transision noise in this function,
            as it will be used for linearizing the system automaticalluy.
        '''
        raise NotImplementedError("The users need to define their own observation method")

    def set_refpoint(self, state=None, input=None, t=None):
        r'''
        Function to set the reference point for linearization.

        Args: 
            state (:obj:`Tensor`): The reference state of the dynamical system. If ``None``,
                the the most recent state is taken. Default: ``None``.
            input (:obj:`Tensor`): The reference input to the dynamical system. If ``None``,
                the the most recent input is taken. Default: ``None``.
            t (:obj:`Tensor`): The reference time step of the dynamical system. If ``None``,
                the the most recent timestamp is taken. Default: ``None``.

        Returns:
            The ``self`` module.

        Warning:
            For nonlinear systems, the users have to call this function before getting the
            linearized system.
        '''
        return self

    @property
    def systime(self):
        r'''
            System time, automatically advanced by :obj:`forward_hook`.
        '''
        return self._t

    @systime.setter
    def systime(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        self._t.copy_(t)


class LTI(System):
    r'''
    Discrete-time Linear Time-Invariant (LTI) system.
    
    Args:
        A (:obj:`Tensor`): The state matrix of LTI system.
        B (:obj:`Tensor`): The input matrix of LTI system.
        C (:obj:`Tensor`): The output matrix of LTI system.
        D (:obj:`Tensor`): The observation matrix of LTI system,
        c1 (:obj:`Tensor`, optional): The constant input of LTI system. Default: ``None``
        c2 (:obj:`Tensor`, optional): The constant output of LTI system. Default: ``None``

    A linear time-invariant lumped system can be described by state-space equation of the form:

    .. math::
        \begin{align*}
            \mathbf{x}_{k+1} = \mathbf{A}\mathbf{x}_k + \mathbf{B}\mathbf{u}_k + \mathbf{c}_1 \\
            \mathbf{y}_k     = \mathbf{C}\mathbf{x}_k + \mathbf{D}\mathbf{u}_k + \mathbf{c}_2 \\
        \end{align*}

    where :math:`\mathbf{x}` and :math:`\mathbf{u}` are state and input of the current
    timestamp of LTI system.

    Note:
        The variables including state and input are row vectors, which is the last dimension of
        a Tensor. :obj:`A`, :obj:`B`, :obj:`C`, :obj:`D`, :obj:`x`, :obj:`u` could be a single
        matrix or batched matrices. In the batch case, their dimensions must be consistent so that
        they can be multiplied for each channel.

    Example:
        >>> # Batch, State, Input, Observe Dimension
        >>> Bd, Sd, Id, Od = 2, 3, 2, 2
        >>> # Linear System Matrices
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> A = torch.randn(Bd, Sd, Sd)
        >>> B = torch.randn(Bd, Sd, Id)
        >>> C = torch.randn(Bd, Od, Sd)
        >>> D = torch.randn(Bd, Od, Id)
        >>> c1 = torch.randn(Bd, Sd)
        >>> c2 = torch.randn(Bd, Od)
        ...
        >>> lti = pp.module.LTI(A, B, C, D, c1, c2).to(device)
        ...
        >>> state = torch.randn(Bd, Sd, device=device)
        >>> input = torch.randn(Bd, Id, device=device)
        >>> state, observation = lti(state, input)
        tensor([[[-8.5639,  0.0523, -0.2576]],
                [[ 4.1013, -1.5452, -0.0233]]]), 
        tensor([[[-3.5780, -2.2970, -2.9314]], 
                [[-0.4358,  1.7306,  2.7514]]]))

    Note:
        In this general example, all variables are in a batch. User definable as appropriate.

    Note:
        More practical examples can be found at `examples/module/dynamics
        <https://github.com/pypose/pypose/tree/main/examples/module/dynamics>`_.
    '''
    
    def __init__(self, A, B, C, D, c1=None, c2=None):
        super().__init__()
        self.register_buffer('_A', A)
        self.register_buffer('_B', B)
        self.register_buffer('_C', C)
        self.register_buffer('_D', D)
        self.register_buffer('_c1', c1)
        self.register_buffer('_c2', c2)

    def forward(self, state, input):
        r'''
        Perform one step advance for the LTI system.

        Args:
            state (:obj:`Tensor`): The state (:math:`\mathbf{x}`) of the system 
            input (:obj:`Tensor`): The input (:math:`\mathbf{u}`) of the system.

        Returns:
            ``tuple`` of Tensor: The state and observation of the system in next time step.
        '''
        return super().forward(state, input)

    def state_transition(self, state, input):
        r'''
        Perform one step of LTI state transition.

        .. math::
            \mathbf{x}_{k+1} = \mathbf{A}\mathbf{x}_k + \mathbf{B}\mathbf{u}_k + \mathbf{c}_1

        Args:
            state (:obj:`Tensor`): The state (:math:`\mathbf{x}`) of the system 
            input (:obj:`Tensor`): The input (:math:`\mathbf{u}`) of the system.

        Returns:
            ``Tensor``: The state the system in next time step.
        '''
        z = bmv(self.A, state) + bmv(self.B, input)
        return z if self.c1 is None else z + self.c1

    def observation(self, state, input):
        r'''
        Return the observation of LTI system.

        .. math::
            \mathbf{y}_k = \mathbf{C}\mathbf{x}_k + \mathbf{D}\mathbf{u}_k + \mathbf{c}_2

        Args:
            state (:obj:`Tensor`): The state (:math:`\mathbf{x}`) of the system 
            input (:obj:`Tensor`): The input (:math:`\mathbf{u}`) of the system.

        Returns:
            ``Tensor``: The observation of the system in next time step.
        '''
        y = bmv(self.C, state) + bmv(self.D, input)
        return y if self.c2 is None else y + self.c2

    @property
    def A(self):
        r'''System transision matrix.'''
        return self._A

    @property
    def B(self):
        r'''System input matrix.'''
        return self._B

    @property
    def C(self):
        r'''System output matrix.'''
        return self._C
 
    @property
    def D(self):
        r'''System observation matrix.'''
        return self._D

    @property
    def c1(self):
        r'''Constant term generated by state-transition.'''
        return self._c1

    @property
    def c2(self):
        r'''Constant term generated by observation.
        '''
        return self._c2


class LTV(LTI):
    r'''
    Discrete-time Linear Time-Variant (LTV) system.
    
    Args:
        A (:obj:`Tensor`, optional): The stacked state matrix of LTI system. Default: ``None``
        B (:obj:`Tensor`, optional): The stacked input matrix of LTI system. Default: ``None``
        C (:obj:`Tensor`, optional): The stacked output matrix of LTI system. Default: ``None``
        D (:obj:`Tensor`, optional): The stacked observation matrix of LTI system. Default: ``None``
        c1 (:obj:`Tensor`, optional): The stacked constant input of LTI system. Default: ``None``
        c2 (:obj:`Tensor`, optional): The stacked constant output of LTI system. Default: ``None``

    A linear time-variant lumped system can be described by state-space equation of the form:

    .. math::
        \begin{align*}
          \mathbf{x}_{k+1} = \mathbf{A}_k\mathbf{x}_k + \mathbf{B}\mathbf{u}_k + \mathbf{c}^1_k \\
          \mathbf{y}_k     = \mathbf{C}_k\mathbf{x}_k + \mathbf{D}\mathbf{u}_k + \mathbf{c}^2_k \\
        \end{align*}

    where :math:`\mathbf{x}_k` and :math:`\mathbf{u}_k` are state and input at the
    timestamp :math:`k` of LTI system.

    Note:
        The :obj:`forward` method implicitly increments the time step via :obj:`forward_hook`.
        :obj:`state_transition` and :obj:`observation` still accept time for the flexiblity
        such as time-varying system. One can directly access the current system time via the
        property :obj:`systime` or :obj:`_t`.

    Note:
        The variables including state and input are row vectors, which is the last dimension of
        a Tensor. :obj:`A`, :obj:`B`, :obj:`C`, :obj:`D`, :obj:`x`, :obj:`u` could be a single
        matrix or batched matrices. In the batch case, their dimensions must be consistent so that
        they can be multiplied for each channel.

    Example:
        A periodic linear time variant system.

        >>> n_batch, n_state, n_ctrl, T = 2, 4, 3, 5
        >>> n_sc = n_state + n_ctrl
        >>> A = torch.randn(n_batch, T, n_state, n_state)
        >>> B = torch.randn(n_batch, T, n_state, n_ctrl)
        >>> C = torch.tile(torch.eye(n_state), (n_batch, T, 1, 1))
        >>> D = torch.tile(torch.zeros(n_state, n_ctrl), (n_batch, T, 1, 1))
        >>> x = torch.randn(n_state)
        >>> u = torch.randn(T, n_ctrl)
        ... 
        >>> class MyLTV(pp.module.LTV):
        ...     def __init__(self, A, B, C, D, T):
        ...         super().__init__(A, B, C, D)
        ...         self.T = T
        ... 
        ...     @property
        ...     def A(self):
        ...         return self._A[...,self._t % self.T,:,:]
        ... 
        ...     @property
        ...     def B(self):
        ...         return self._B[...,self._t % self.T,:,:]
        ... 
        ...     @property
        ...     def C(self):
        ...         return self._C[...,self._t % self.T,:,:]
        ... 
        ...     @property
        ...     def D(self):
        ...         return self._D[...,self._t % self.T,:,:]
        ... 
        >>> ltv = MyLTV(A, B, C, D, T)
        >>> for t in range(T):
        ...     x, y = ltv(x, u[t])

        One may also generate the system matrices with the time variable :obj:`_t`.

        >>> n_batch, n_state, n_ctrl, T = 2, 4, 3, 5
        >>> n_sc = n_state + n_ctrl
        >>> x = torch.randn(n_state)
        >>> u = torch.randn(T, n_ctrl)
        ... 
        >>> class MyLTV(pp.module.LTV):
        ...     def __init__(self, A, B, C, D, T):
        ...         super().__init__(A, B, C, D)
        ...         self.T = T
        ... 
        ...     @property
        ...     def A(self):
        ...         return torch.eye(4, 4) * self._t.cos()
        ... 
        ...     @property
        ...     def B(self):
        ...         return torch.eye(4, 3) * self._t.sin()
        ... 
        ...     @property
        ...     def C(self):
        ...         return torch.eye(4, 4) * self._t.tan()
        ... 
        ...     @property
        ...     def D(self):
        ...         return torch.eye(4, 3)
        ... 
        >>> ltv = MyLTV()
        >>> for t in range(T):
        ...     x, y = ltv(x, u[t])

    Note:
        More practical examples can be found at `examples/module/dynamics
        <https://github.com/pypose/pypose/tree/main/examples/module/dynamics>`_.
    '''
    def __init__(self, A=None, B=None, C=None, D=None, c1=None, c2=None):
        super().__init__(A, B, C, D, c1, c2)

    def set_refpoint(self, state=None, input=None, t=None):
        r'''
        Function to set the reference point for linearization.

        Args: 
            state (:obj:`Tensor`): The reference state of the dynamical system. If ``None``,
                the the most recent state is taken. Default: ``None``.
            input (:obj:`Tensor`): The reference input to the dynamical system. If ``None``,
                the the most recent input is taken. Default: ``None``.
            t (:obj:`Tensor`): The reference time step of the dynamical system. If ``None``,
                the the most recent timestamp is taken. Default: ``None``.

        Returns:
            The ``self`` module.

        Warning:
            For nonlinear systems, the users have to call this function before getting the
            linearized system.
        '''
        self.systime = t
        return self


class NLS(System):
    r'''
    Dynamics model for discrete-time non-linear system (NLS).
    
    The state transision function :math:`\mathbf{f}` and observation function
    :math:`\mathbf{g}` are given by:

    .. math::
        \begin{aligned}
            \mathbf{x}_{k+1} &= \mathbf{f}(\mathbf{x}_k, \mathbf{u}_k, t_k), \\
            \mathbf{y}_{k}   &= \mathbf{g}(\mathbf{x}_k, \mathbf{u}_k, t_k), 
        \end{aligned}

    where :math:`k`, :math:`\mathbf{x}`, :math:`\mathbf{u}`, :math:`\mathbf{y}` are the time
    step, state(s), input(s), and observation(s), respectively.

    Note:
        To use the class, users need to inherit this class and define methods
        :obj:`state_transition` and :obj:`observation`, which are automatically called by
        internal :obj:`forward` method.
        The system timestamp (starting from **0**) is also self-added automatically once
        the :obj:`forward` method is called.

    Note:

        This class provides automatic **linearlization** at a reference point
        :math:`\chi^*=(\mathbf{x}^*, \mathbf{u}^*, t^*)` along a trajectory.
        One can directly call those linearized system matrices as properties including
        :obj:`A`, :obj:`B`, :obj:`C`, :obj:`D`, :obj:`c1`, and :obj:`c2`, after calling
        a method :obj:`set_refpoint`.

        Consider a point
        :math:`\chi=(\mathbf{x}^*+\delta\mathbf{x}, \mathbf{u}^*+\delta\mathbf{u}, t^*)` near
        :math:`\chi^*`. We have

        .. math::
            \begin{aligned}
            \mathbf{f}(\mathbf{x}, \mathbf{u}, t^*) &\approx \mathbf{f}(\mathbf{x}^*, 
                \mathbf{u}^*, t^*) +  \left. \frac{\partial \mathbf{f}}{\partial \mathbf{x}}
                \right|_{\chi^*} \delta \mathbf{x} + \left. \frac{\partial \mathbf{f}} 
                {\partial \mathbf{u}} \right|_{\chi^*} \delta \mathbf{u} \\
            &= \mathbf{f}(\mathbf{x}^*, \mathbf{u}^*, t^*) + \mathbf{A}(\mathbf{x} 
                - \mathbf{x}^*) + \mathbf{B}(\mathbf{u}-\mathbf{u}^*) \\
            &= \mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{u} + \mathbf{c}_1
            \end{aligned}

        and

        .. math::
            \mathbf{g}(\mathbf{x}, \mathbf{u}, t^*) \approx \mathbf{C}\mathbf{x} \
                        + \mathbf{D}\mathbf{u} + \mathbf{c}_2

        The notion of linearization is slightly different from that in dynamical system
        theory. First, the linearization can be done for arbitrary point(s), not limited to
        the system's equilibrium point(s), and therefore the extra constant terms :math:`\mathbf{c}_1`
        and :math:`\mathbf{c}_2` are produced. Second, the linearized equations are represented
        by the full states and inputs: :math:`\mathbf{x}` and :math:`\mathbf{u}`, rather than 
        the perturbation format: :math:`\delta \mathbf{x}` and :math:`\delta \mathbf{u}`
        so that the model is consistent with, e.g., the LTI model and the iterative LQR
        solver. For more details go to :meth:`LTI`.

    Example:

        A simple linear time-varying system, but defined via NLS. Here we show an example
        for advancing one time step of the system at a given time step and computing the
        linearization.

        >>> import math, torch
        >>> import pypose as pp
        ... 
        >>> class Floquet(pp.module.NLS):
        ...     def __init__(self):
        ...         super().__init__()
        ... 
        ...     def state_transition(self, state, input, t):
        ... 
        ...         cc = (2 * math.pi * t / 100).cos()
        ...         ss = (2 * math.pi * t / 100).sin()
        ... 
        ...         A = torch.tensor([[   1., cc/10],
        ...                           [cc/10,    1.]], device=t.device)
        ...         B = torch.tensor([[ss],
        ...                           [1.]], device=t.device)
        ... 
        ...         return pp.bmv(A, state) + pp.bmv(B, input)
        ... 
        ...     def observation(self, state, input, t):
        ...         return state + t
        ...
        >>> # Start from t = 8, and advance one step to t = 9.
        >>> step, current = 8, torch.tensor([1., 1.])
        >>> input = torch.tensor(2 * math.pi / 50 * step).sin()
        ... 
        >>> system = Floquet().reset(t = step)
        >>> next, observation = system(current, input)
        >>> system.set_refpoint()
        ... 
        >>> print(next)        # Next state
        >>> print(observation) # Observation
        >>> print(system.A)    # Linearized state matrix
        >>> print(system.B)    # Linearized input matrix
        tensor([1.4944, 1.9320])
        tensor([9., 9.])
        tensor([[1.0000, 0.0844],
                [0.0844, 1.0000]])
        tensor([[0.5358],
                [1.0000]])

    Note:
        For generating one trajecotry given a series of inputs, advanced use of
        linearization, and more practical examples can be found at `examples/module/dynamics
        <https://github.com/pypose/pypose/tree/main/examples/module/dynamics>`_.
    '''
    def __init__(self):
        super().__init__()
        self.jacargs = {'vectorize':True, 'strategy':'reverse-mode'}

    def forward(self, state, input):
        r'''
        Defines the computation performed at every call that advances the system by one time step.

        Note:
            The :obj:`forward` method implicitly increments the time step via :obj:`forward_hook`.
            :obj:`state_transition` and :obj:`observation` still accept time for the flexiblity
            such as time-varying system. One can directly access the current system time via the
            property :obj:`systime`.

        Note:
            To introduce noise in a model, redefine this method via
            subclassing. See example in ``examples/module/ekf/tank_robot.py``.
        '''
        self.state, self.input = torch.atleast_1d(state), torch.atleast_1d(input)
        state = self.state_transition(self.state, self.input, self.systime)
        obs = self.observation(self.state, self.input, self.systime)
        return state, obs

    def set_refpoint(self, state=None, input=None, t=None):
        r'''
        Function to set the reference point for linearization.

        Args: 
            state (:obj:`Tensor`): The reference state of the dynamical system. If ``None``,
                the the most recent state is taken. Default: ``None``.
            input (:obj:`Tensor`): The reference input to the dynamical system. If ``None``,
                the the most recent input is taken. Default: ``None``.
            t (:obj:`Tensor`): The reference time step of the dynamical system. If ``None``,
                the the most recent timestamp is taken. Default: ``None``.

        Returns:
            The ``self`` module.

        Warning:
            For nonlinear systems, the users have to call this function before getting the
            linearized system.
        '''
        self._ref_state = self.state if state is None else torch.atleast_1d(state)
        self._ref_input = self.input if input is None else torch.atleast_1d(input)
        self._ref_t = self.systime if t is None else torch.atleast_1d(t)
        self._ref_f = self.state_transition(self._ref_state, self._ref_input, self._ref_t)
        self._ref_g = self.observation(self._ref_state, self._ref_input, self._ref_t)
        return self

    @property
    def A(self):
        r'''
        Linear/linearized system state matrix.

        .. math::
            \mathbf{A} = \left. \frac{\partial \mathbf{f}}{\partial \mathbf{x}} \right|_{\chi^*}
        '''
        func = lambda x: self.state_transition(x, self._ref_input, self._ref_t)
        return jacobian(func, self._ref_state, **self.jacargs)

    @property
    def B(self):
        r'''
        Linear/linearized system input matrix.

        .. math::
            \mathbf{B} = \left. \frac{\partial \mathbf{f}}{\partial \mathbf{u}} \right|_{\chi^*}
        '''
        func = lambda x: self.state_transition(self._ref_state, x, self._ref_t)
        return jacobian(func, self._ref_input, **self.jacargs)

    @property
    def C(self):
        r'''
        Linear/linearized system output matrix.

        .. math::
            \mathbf{C} = \left. \frac{\partial \mathbf{g}}{\partial \mathbf{x}} \right|_{\chi^*}
        '''
        func = lambda x: self.observation(x, self._ref_input, self._ref_t)
        return jacobian(func, self._ref_state, **self.jacargs)
 
    @property
    def D(self):
        r'''
        Linear/Linearized system observation matrix.

        .. math::
            \mathbf{D} = \left. \frac{\partial \mathbf{g}}
                                {\partial \mathbf{u}} \right|_{\chi^*}
        '''
        func = lambda x: self.observation(self._ref_state, x, self._ref_t)
        return jacobian(func, self._ref_input, **self.jacargs)

    @property
    def c1(self):
        r'''
        Constant term generated by state-transition.

        .. math::
            \mathbf{c}_1 = \mathbf{f}(\mathbf{x}^*, \mathbf{u}^*, t^*)
                           - \mathbf{A}\mathbf{x}^* - \mathbf{B}\mathbf{u}^*
        '''
        # Potential performance loss here - self.A and self.B involves jacobian eval
        return self._ref_f - bmv(self.A, self._ref_state) - bmv(self.B, self._ref_input)

    @property
    def c2(self):
        r'''
        Constant term generated by observation.

        .. math::
            \mathbf{c}_2 = \mathbf{g}(\mathbf{x}^*, \mathbf{u}^*, t^*)
                           - \mathbf{C}\mathbf{x}^* - \mathbf{D}\mathbf{u}^*
        '''
        # Potential performance loss here - self.C and self.D involves jacobian eval
        return self._ref_g - bmv(self.C, self._ref_state) - bmv(self.D, self._ref_input)
