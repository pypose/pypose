import torch
from .. import bmv
from torch import nn
from ..func import jacrev

class System(nn.Module):
    r'''
    The base class for a system dynamics model.

    In most of the cases, users only need to subclass a specific dynamic system,
    such as linear time invariant system :meth:`LTI`, Linear Time-Variant :meth:`LTV`,
    and a non-linear system :meth:`NLS`.
    '''
    def __init__(self):
        super().__init__()
        self._ref_state, self._ref_input, self._ref_t = None, None, None


    def forward(self, state, input, t=None):
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
        self.set_refpoint(state, input, t)
        state = self.state_transition(self.state, self.input, t)# should be with t
        obs = self.observation(self.state, self.input, t)
        return state, obs

    def _register_attr(self, name, attr):
        if isinstance(attr,torch.Tensor):
            self.register_buffer(name,attr)
        elif isinstance(attr,nn.Module):
            self.register_module(name,attr)
        elif attr is None:
            self.register_parameter(name,None)
        else:
            raise TypeError("The attribute must be a Tensor or a Module")

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
        if t is None:
            pass
        else:
            t=torch.tensor(t)

        self._ref_state, self._ref_input, self._ref_t = state, input, t


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
        self._register_attr('A', A)
        self._register_attr('B', B)
        self._register_attr('C', C)
        self._register_attr('D', D)
        self._register_attr('c1', c1)
        self._register_attr('c2', c2)

    def state_transition(self, state, input, t=None):
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

    def observation(self, state, input, t):
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
    def c1(self):
        r'''Constant term generated by state-transition.'''
        return self._c1

    @property
    def c2(self):
        r'''Constant term generated by observation.
        '''
        return self._c2


class LTV(System):
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
    def __init__(self,):
        super().__init__()
        self._c1, self._c2 = None, None

    def A(self,t):
        r'''
        The state matrix getter of LTV system.

        Args:
            t (:obj:`Tensor`): The time step of the dynamical system.

        Returns:
            ``Tensor``: The state matrix of the system at the current step.
        '''
        raise NotImplementedError("The users need to define their own state transition method")


    def B(self,t):
        r'''
        The state matrix getter of LTV system.

        Args:
            t (:obj:`Tensor`): The time step of the dynamical system.

        Returns:
            ``Tensor``: The state matrix of the system at the current step.
        '''
        raise NotImplementedError("The users need to define their own state transition method")


    def C(self,t):
        r'''
        The state matrix getter of LTV system.

        Args:
            t (:obj:`Tensor`): The time step of the dynamical system.

        Returns:
            ``Tensor``: The state matrix of the system at the current step.
        '''
        raise NotImplementedError("The users need to define their own state transition method")


    def D(self,t):
        r'''
        The state matrix getter of LTV system.

        Args:
            t (:obj:`Tensor`): The time step of the dynamical system.

        Returns:
            ``Tensor``: The state matrix of the system at the current step.
        '''
        raise NotImplementedError("The users need to define their own state transition method")

    @property
    def c1(self):
        r'''Constant term generated by state-transition.'''
        return self._c1

    @property
    def c2(self):
        r'''Constant term generated by observation.
        '''
        return self._c2

    def state_transition(self, state, input, t=None):
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
        z = bmv(self.A(t), state) + bmv(self.B(t), input)
        return z if self.c1 is None else z + self.c1

    def observation(self, state, input, t):
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
        y = bmv(self.C(t), state) + bmv(self.D(t), input)
        return y if self.c2 is None else y + self.c2


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

    def c1(self, state, input, t):

        f = self.state_transition(state, input,t)
        A, B = sysmat(self, state, input, t, 'AB')
        return f - bmv(A, state) - bmv(B, input)

    def c2(self, state, input, t):

        g = self.observation(state, input,t)
        C, D = sysmat(self, state, input, t, 'CD')
        return g - bmv(C, state) - bmv(D, input)


def toBTN(vec, T):
    r'''
    Reshape the input tensor of shape ``[..., N]`` to ``[B, T, N]``, where B, T, N
    normally refer to the dimension of batch, time step, and state, respectively.

    Returns:
        The reshaped tensor in shape of ``[B, T, N]``.
    '''
    if vec.ndim == 1:
        vec = vec.unsqueeze(0)

    if vec.ndim == 2:
        vec = vec.unsqueeze(0)

    if vec.shape[1] == 1:
        vec = vec.repeat(1, T, 1)

    return vec


def runsys(system: System, T, x_traj, u_traj):
    r'''
    Run the system for T steps, given state and input trajectories or vectors.

    Returns:
        The state trajectory of the system based on the state and input trajectories.
    '''

    #make initial states trajectories if not given
    x_traj = toBTN(x_traj, T)
    u_traj = toBTN(u_traj, T)

    for i in range(T-1):
        x_traj[...,i+1,:], _ = system(x_traj[...,i,:], u_traj[...,i,:], i)

    return x_traj


def sysmat(system:System, state, input, t, mats):
    r'''
    Find the linearized system matrices at given states, inputs, and time.
    Argument tensors state, input, and t should be batched.

    Returns:
        Matrices A, B, C, D of the linearized system.
    '''

    mats = mats.upper()

    #get total number of time steps

    if t is None:
        T = 1
    elif isinstance(t,int) or t.ndim==0:
        T = int(t)
    else:
        T = t.shape[-1]

    if isinstance(system, LTV):
        dict = {'A':system.A, 'B':system.B, 'C':system.C, 'D':system.D}
        return [dict[m](t) for m in mats]

    if isinstance(system, LTI):
        dict = {'A':system.A, 'B':system.B, 'C':system.C, 'D':system.D}
        return [dict[m] if T==1 else dict[m].unsqueeze(1).repeat(1,T,1,1) for m in mats]

    if isinstance(system, NLS):

        funcs = {'A': lambda x: system.state_transition(x, input, t),
                 'B': lambda x: system.state_transition(state, x, t),
                 'C': lambda x: system.observation(x, input, t),
                 'D': lambda x: system.observation(state, x, t)}

        jac_inputs = {'A': state, 'B': input, 'C': state, 'D': input}

        res = []

        for m in mats:
            jacFunc = jacrev(funcs[m])
            M = jacFunc(jac_inputs[m])
            res.append(M)

        return res

    raise TypeError("System type not recognized")
