import torch as torch
import torch.nn as nn
import pypose as pp
from torch.autograd.functional import jacobian


class System(nn.Module):
    r'''
    The general base class of discrete system dynamics model.
    
    The state transision function :math:`\mathbf{f}` and observation functions
    :math:`\mathbf{g}` are given by:

    .. math::
        \begin{aligned}
            \mathbf{x}_{k+1} &= \mathbf{f}(\mathbf{x}_k, \mathbf{u}_k, t_k), \\
            \mathbf{y}_{k}   &= \mathbf{g}(\mathbf{x}_k, \mathbf{u}_k, t_k), 
        \end{aligned}

    where :math:`k`, :math:`\mathbf{x}`, :math:`\mathbf{u}`, :math:`\mathbf{y}` are the time
    step, states, inputs, and observations, respectively.

    Args:
        time (:obj:`bool`): Whether the system is time-varying. Default: ``False``.

    Note:
        Here we choose to work with a time-discrete system, since typically in numerical
        simulations and control applications, the dynamical systems are usually discretized
        in time.

    To use the class, the user should define two methods :obj:`state_transition` and
    :obj:`observation`, which are used in the :obj:`forward` method. The :obj:`forward`
    method advances the dynamical system by one time step, and is the function that is
    back-propagated in the learning process.

    Note:

        **Linearization**

        This class provides a means to linearize the system at a reference point
        :math:`\chi^*=(\mathbf{x}^*, \mathbf{u}^*, t^*)` along a trajectory. Consider a point
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

        One can directly call those properties :obj:`A`, :obj:`B`, :obj:`C`, :obj:`D`, 
        :obj:`c1`, and :obj:`c2`.

        The notion of linearization is slightly different from that in dynamical system
        theory. First, the linearization is done for any arbitrary point, not just at
        the equilibrium point(s), and therefore the extra constant terms :math:`\mathbf{c}_1`
        and :math:`\mathbf{c}_2` are produced. Second, the linearized equations are written
        for the full states and inputs, :math:`\mathbf{x}` and :math:`\mathbf{u}`, instead
        of their perturbation terms, :math:`\delta \mathbf{x}` and :math:`\delta \mathbf{u}`
        so that the model is consistent with, e.g., the LTI model and the iterative LQR
        solver.
    '''

    def __init__(self, time=False):
        super().__init__()
        self.jacargs = {'vectorize':True, 'strategy':'reverse-mode'}
        if time:
            self.register_buffer('t', torch.zeros(1))
            self.register_forward_hook(self.forward_hook)

    def forward_hook(self, module, inputs, outputs):
        self.state, self.input = inputs
        self.t.add_(1)

    def forward(self, state, input, time=None):
        r'''
        Defines the computation performed at every call.

        Should be overridden by all subclasses.

        Note:
            The :obj:`forward` method implicitly increments the time step via :obj:`forward_hook`.
            As a result, in the implementations of :obj:`state_transition` and :obj:`observation`, 
            one does not need to include the time as the input; instead, one can directly access
            the current time via the data member :obj:`self.t`.
        '''
        return self.state_transition(state, input), self.observation(state, input)

    def state_transition(self, state, input, time=None):
        r'''
        Args:
            state : Tensor
                    The state of the dynamical system
            input : Tensor
                    The input to the dynamical system

        Returns:
            Tensor: The state of the system at next time step

        Note:
            The users need to define this method and can access the current time via the data
            member :obj:`self.t`.
        '''
        raise NotImplementedError("The users need to define their own state transition method")

    def observation(self, state, input, time=None):
        r'''
        Args:
            state : Tensor
                    The state of the dynamical system
            input : Tensor
                    The input to the dynamical system

        Returns:
            Tensor: The observation of the system at the current step

        Note:
            The users need to define this method and can access the current time via the data
            member :obj:`self.t`.
        '''
        raise NotImplementedError("The users need to define their own observation method")

    def reset(self, t=0):
        self.t.fill_(t)

    def set_linearization_point(self, state, input, t):
        r'''
        Function to set the point about which the system is to be linearized.

        Parameters
        ----------
        state : Tensor
                The state of the dynamical system
        input : Tensor
                The input to the dynamical system
        t     : Tensor
                The time step of the dynamical system

        Returns
        ----------
        None
        '''
        self.state, self.input, self.t = state, input, t

    @property
    def A(self):
        r'''
        Linear/linearized system state matrix.

        .. math::
            \mathbf{A} = \left. \frac{\partial \mathbf{f}}{\partial \mathbf{x}} \right|_{\chi^*}

        '''
        if hasattr(self, '_A'):
            return self._A
        func = lambda x: self.state_transition(x, self.input)
        return jacobian(func, self.state, **self.jacargs)

    @A.setter
    def A(self, A):
        self._A = A

    @property
    def B(self):
        r'''
        Linear/linearized system input matrix.

        .. math::
            \mathbf{B} = \left. \frac{\partial \mathbf{f}}{\partial \mathbf{u}} \right|_{\chi^*}
        '''
        if hasattr(self, '_B'):
            return self._B
        func = lambda x: self.state_transition(self.state, x)
        return jacobian(func, self.input, **self.jacargs)

    @B.setter
    def B(self, B):
        self._B = B

    @property
    def C(self):
        r'''
        Linear/linearized system output matrix.

        .. math::
            \mathbf{C} = \left. \frac{\partial \mathbf{g}}{\partial \mathbf{x}} \right|_{\chi^*}
        '''
        if hasattr(self, '_C'):
            return self._C
        func = lambda x: self.observation(x, self.input)
        return jacobian(func, self.state, **self.jacargs)

    @C.setter
    def C(self, C):
        self._C = C
 
    @property
    def D(self):
        r'''
        Linear/Linearized system observation matrix.

        .. math::
            \mathbf{D} = \left. \frac{\partial \mathbf{g}}
                                {\partial \mathbf{u}} \right|_{\chi^*}
        '''
        if hasattr(self, '_D'):
            return self._D
        func = lambda x: self.observation(self.state, x)
        return jacobian(func, self.input, **self.jacargs)

    @D.setter
    def D(self, D):
        self._D = D

    @property
    def c1(self):
        r'''
        Constant term generated by state-transition.

        .. math::
            \mathbf{c}_1 = \mathbf{f}(\mathbf{x}^*, \mathbf{u}^*, t^*)
                           - \mathbf{A}\mathbf{x}^* - \mathbf{B}\mathbf{u}^*
        '''
        if hasattr(self, '_c1'):
            return self._c1
        return self.state_transition(self.state, self.input) \
            - (self.state).matmul(self.A.mT) - (self.input).matmul(self.B.mT)

    @c1.setter
    def c1(self, c1):
        self._c1 = c1
    
    @property
    def c2(self):
        r'''
        Constant term generated by observation.

        .. math::
            \mathbf{c}_2 = \mathbf{g}(\mathbf{x}^*, \mathbf{u}^*, t^*)
                           - \mathbf{C}\mathbf{x}^* - \mathbf{D}\mathbf{u}^*
        '''
        if hasattr(self, '_c2'):
            return self._c2
        return self.observation(self.state, self.input) \
            - (self.state).matmul(self.C.mT) - (self.input).matmul(self.D.mT)

    @c2.setter
    def c2(self, c2):
        self._c2 = c2


class LTI(System):
    r'''
    Discrete-time Linear Time-Invariant (LTI) system.
    
    Args:
        A, B, C, D (:obj:`Tensor`): The coefficient matrix in the state-space equation of LTI system, 
        c1 (:obj:`Tensor`): The constant input of the system, 
        c2 (:obj:`Tensor`): The constant output of the system, 
        state (:obj:`Tensor`): The state of the current timestamp of LTI system, 
        input (:obj:`Tensor`): The input of the current timestamp of LTI system.

    Return:
        Tuple of Tensors: The state of the next timestamp (state-transition) and the system output (observation).

    Every linear time-invariant lumped system can be described by a set of equations of the form 
    which is called the state-space equation.

    .. math::
        \begin{align*}
            \mathbf{z} = \mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{u} + \mathbf{c}_1 \\
            \mathbf{y} = \mathbf{C}\mathbf{x} + \mathbf{D}\mathbf{u} + \mathbf{c}_2 \\
        \end{align*}

    where we use :math:`\mathbf{x}` and :math:`\mathbf{u}` to represent state and input of the current timestamp of LTI system.
            
    Here, we consider the discrete-time system dynamics.  
        
    Note:
        According to the actual physical meaning, the dimensions of A, B, C, D must be the consistent, 
        whether in batch or not.

        :math:`\mathbf{A}`, :math:`\mathbf{B}`, :math:`\mathbf{C}`, :math:`\mathbf{D}`, :math:`\mathbf{x}`, :math:`\mathbf{u}` 
        could be a single input or in a batch. In the batch case, their dimensions must be consistent 
        so that they can be multiplied for each channel.
             
        Note that here variables are given as row vectors.
    '''
    
    def __init__(self, A, B, C, D, c1=None, c2=None):
        super(LTI, self).__init__(time=False)
        assert A.ndim == B.ndim == C.ndim == D.ndim, "Invalid System Matrices dimensions"
        self.A, self.B, self.C, self.D = A, B, C, D
        self.c1, self.c2 = c1, c2
    
    def forward(self, state, input):
        r'''

        Example:
            >>> A = torch.randn((2, 3, 3))
                B = torch.randn((2, 3, 2))
                C = torch.randn((2, 3, 3))
                D = torch.randn((2, 3, 2))
                c1 = torch.randn((2, 1, 3))
                c2 = torch.randn((2, 1, 3))
                state = torch.randn((2, 1, 3))
                input = torch.randn((2, 1, 2))
            >>> lti = pp.module.LTI(A, B, C, D, c1, c2)
            >>> lti(state, input)
            tensor([[[-8.5639,  0.0523, -0.2576]], 
                    [[ 4.1013, -1.5452, -0.0233]]]), 
            tensor([[[-3.5780, -2.2970, -2.9314]], 
                    [[-0.4358,  1.7306,  2.7514]]]))

        Note:
            In this general example, all variables are in a batch. User definable as appropriate.
        '''
        if self.A.ndim >= 3:
            assert self.A.ndim == state.ndim == input.ndim,  "Invalid System Matrices dimensions"
        else:
            assert self.A.ndim == 2,  "Invalid System Matrices dimensions"

        z = self.state_transition(state, input)
        y = self.observation(state, input)
        return z, y

    def state_transition(self, state, input):
        return state.matmul(self.A.mT) + input.matmul(self.B.mT) + self.c1

    def observation(self, state, input):
        return state.matmul(self.C.mT) + input.matmul(self.D.mT) + self.c2