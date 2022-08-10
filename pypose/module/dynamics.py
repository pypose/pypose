from numpy import vectorize
import torch as torch
import torch.nn as nn
from torch.autograd.functional import jacobian

class _System(nn.Module):
    r'''
    A sub-class of :obj:`torch.nn.Module` for the simulation and linearization of nonlinear
    time-discrete time-varying dynamics.

    Args:
        time (:obj:`boolean`): Whether the system is time-varying; defaults to False, meaning time-invariant

    Governing Equation
    ----------
    The nonlinear state-space equation is given as:

    .. math::
        \begin{align}
        \mathbf{x}_{k+1} &= \mathbf{f}(\mathbf{x}_k,\mathbf{u}_k,t_k) \\
        \mathbf{y}_{k} &= \mathbf{g}(\mathbf{x}_k,\mathbf{u}_k,t_k)
        \end{align}

    where :math:`k` is the time step; :math:`\mathbf{x}\in\mathbb{R}^n`, :math:`\mathbf{u}\in\mathbb{R}^m`,
    :math:`\mathbf{y}\in\mathbb{R}^p` are the states, inputs and observations, respectively;
    :math:`\mathbf{f}:\mathbb{R}^n\times\mathbb{R}^m\times\mathbb{R}\mapsto\mathbb{R}^n` and
    :math:`\mathbf{g}:\mathbb{R}^n\times\mathbb{R}^m\times\mathbb{R}\mapsto\mathbb{R}^p` are
    the state transition and observation functions, respectively.

    Note:
        Here we choose to work with a time-discrete system, since typically in numerical simulations and
        control applications, the dynamical systems are usually discretized in time.  However, in future,
        the continuous dynamical will also be considered and implemented.

    To use the class, the user should define two methods `state_transition` and `observation`, which are
    used in the `forward` method.  The `forward` method advances the dynamical system by one time step, and
    is the function that is back-propagated in the learning process.

    Note:
        The `forward` method implicitly increments the time step via `forward_hook`.  As a result, in the
        implementations of `state_transition` and `observation`, one does not need to include the time as
        the input; instead, one can directly access the current time via the data member `self.t`.

    Linearization
    ----------
    This class provides a means to linearize the system at a reference point :math:`\chi^*=(\mathbf{x}^*,\mathbf{u}^*,t^*)`
    along a trajectory.  Consider a point :math:`\chi=(\mathbf{x}^*+\delta\mathbf{x},\mathbf{u}^*+\delta\mathbf{u},t^*)`
    near :math:`\chi^*`.  The goal is to obtain a linear dynamics model for :math:`\chi` that behaves
    similarly to the nonlinear dynamics when :math:`\delta\mathbf{x}||` and :math:`\delta\mathbf{u}||` are sufficiently
    small.

    The linearization process is the same for :math:`\mathbf{f}` and :math:`\mathbf{g}`, so we only present
    the case for :math:`\mathbf{f}` in detail below.
    Ignoring the higher order terms, the Taylor series expansion of :math:`\mathbf{f}` at the reference point
    :math:`\chi^*` is

    .. math::
        \begin{aligned}
        \mathbf{f}(\mathbf{x},\mathbf{u},t^*) &\approx \mathbf{f}(\mathbf{x}^*,\mathbf{u}^*,t^*) + 
                                \left. \frac{\partial \mathbf{f}}{\partial \mathbf{x}} \right|_{\chi^*} \delta \mathbf{x} +
                                \left. \frac{\partial \mathbf{f}}{\partial \mathbf{u}} \right|_{\chi^*} \delta \mathbf{u} \\
        &= \mathbf{f}(\mathbf{x}^*,\mathbf{u}^*,t^*) + \mathbf{A}(\mathbf{x}-\mathbf{x}^*) + \mathbf{B}(\mathbf{u}-\mathbf{u}^*) \\
        &= \mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{u} + \mathbf{c}_1
        \end{aligned}

    where we have introduced,

    .. math::
        \mathbf{A} = \left. \frac{\partial \mathbf{f}}{\partial \mathbf{x}} \right|_{\chi^*},\quad 
        \mathbf{B} = \left. \frac{\partial \mathbf{f}}{\partial \mathbf{u}} \right|_{\chi^*},\quad
        \mathbf{c}_1 = \mathbf{f}(\mathbf{x}^*,\mathbf{u}^*,t^*) - \mathbf{A}\mathbf{x}^* - \mathbf{B}\mathbf{u}^*

    Similarly, one can linearize :math:`\mathbf{g}` as

    .. math::
        \mathbf{g}(\mathbf{x},\mathbf{u},t^*) \approx \mathbf{C}\mathbf{x} + \mathbf{D}\mathbf{u} + \mathbf{c}_2

    where

    .. math::
        \mathbf{C} = \left. \frac{\partial \mathbf{g}}{\partial \mathbf{x}} \right|_{\chi^*},\quad 
        \mathbf{D} = \left. \frac{\partial \mathbf{g}}{\partial \mathbf{u}} \right|_{\chi^*},\quad
        \mathbf{c}_2 = \mathbf{g}(\mathbf{x}^*,\mathbf{u}^*,t^*) - \mathbf{C}\mathbf{x}^* - \mathbf{D}\mathbf{u}^*

    Note: 
        The notion of linearization is slightly different from that in dynamical system theory.
        First, the linearization is done for any arbitrary point, not just at the equilibrium point(s), and
        therefore the extra ``bias'' terms :math:`\mathbf{c}_1` and :math:`\mathbf{c}_2` are produced.
        Second, the linearized equations are written for the full states and inputs, :math:`\mathbf{x}` and :math:`\mathbf{u}`,
        instead of their perturbation terms, :math:`\delta \mathbf{x}` and :math:`\delta \mathbf{u}`
        so that the model is consistent with, e.g., the `LTI` model and the iterative LQR solver.
    '''

    def __init__(self, time=False):
        super().__init__()
        self.jacargs = {'vectorize':True, 'strategy':'reverse-mode'}
        if time:
            self.register_buffer('t',torch.zeros(1))
            self.register_forward_hook(self.forward_hook)

    def forward_hook(self, module, inputs, outputs):
        self.input, self.state = inputs
        self.t.add_(1)

    def forward(self, state, input):
        r'''
        Parameters
        ----------
        state : Tensor
                The state of the dynamical system
        input : Tensor
                The input to the dynamical system

        Returns
        -------
        new_state   : Tensor
                      The state of the system at next time step
        observation : Tensor
                      The observation of the system at the current step
        '''

        new_state = self.state_transition(state, input)
        observation = self.observation(state, input)
        return new_state, observation

    def state_transition(self, state, input):
        r'''
        Parameters
        ----------
        state : Tensor
                The state of the dynamical system
        input : Tensor
                The input to the dynamical system
        
        Note:
            One can access the current time via the data member `self.t`.

        Returns
        ----------
        new_state   : Tensor
                      The state of the system at next time step
        '''
        raise NotImplementedError("The users need to define their own state transition method")

    def observation(self, state, input):
        r'''
        Parameters
        ----------
        state : Tensor
                The state of the dynamical system
        input : Tensor
                The input to the dynamical system
        
        Note:
            One can access the current time via the data member `self.t`.

        Returns
        ----------
        observation : Tensor
                      The observation of the system at the current step
        '''
        raise NotImplementedError("The users need to define their own observation method")

    def reset(self,t=0):
        self.t.fill_(0) 

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
        State matrix for linear/linearized system (:math:`\mathbf{A}`)
        '''
        if hasattr(self, '_A'):
            return self._A
        func = lambda x: self.state_transition(x, self.input)
        self._A = jacobian(func, self.state, **self.jacargs)
        return self._A

    @property
    def B(self):
        r'''
        Input matrix for linear/linearized system (:math:`\mathbf{B}`)
        '''
        if hasattr(self, '_B'):
            return self._B
        func = lambda x: self.state_transition(self.state, x)
        self._B = jacobian(func, self.input, **self.jacargs)
        return self._B

    @property
    def C(self):
        r'''
        Output matrix for linear/linearized system (:math:`\mathbf{C}`)
        '''
        if hasattr(self, '_C'):
            return self._C
        func = lambda x: self.observation(x, self.input)
        self._C = jacobian(func, self.state, **self.jacargs)
        return self._C

    @property
    def D(self):
        r'''
        Feedthrough matrix for linear/linearized system (:math:`\mathbf{D}`)
        '''
        if hasattr(self, '_D'):
            return self._D
        func = lambda x: self.observation(self.state, x)
        self._D = jacobian(func, self.input, **self.jacargs)
        return self._D
    
    @property
    def c1(self):
        r'''
        Bias generated by state-transition (:math:`c_1`)
        '''
        if hasattr(self,'_c1'):
            return self._c1
        self._c1 = self.state_transition(self.state,self.input)-(self.state).matmul(self.A.mT)-(self.input).matmul(self.B.mT)
        return self._c1

    @property
    def c2(self):
        r'''
        Bias generated by observation (:math:`c_2`)
        '''
        if hasattr(self,'_c2'):
            return self._c2
        self._c2 = self.observation(self.state,self.input)-(self.state).matmul(self.C.mT)-(self.input).matmul(self.D.mT)
        return self._c2

class LTI(_System):
    r'''
    A sub-class of: obj: '_System' to represent Linear Time-Invariant system.
    
    Args:
        A, B, C, D (:obj:`Tensor`): The input tensor in the state-space equation of LTI system,
            usually in matrix form.
        c1, c2 (:obj:`Tensor`): Bias generated by system.
        
    Note:
        According to the actual physical meaning, the dimensions of A, B, C, D must be the same,
        whether in the batch case or not.
        
        The system is time invariant.
    '''
    def __init__(self, A, B, C, D, c1=None, c2=None):
        super(LTI, self).__init__(time=False)
        assert A.ndim == B.ndim == C.ndim == D.ndim, "Invalid System Matrices dimensions"
        self._A, self._B, self._C, self._D = A, B, C, D
        self._c1, self._c2 = c1, c2

    @property
    def c1(self):
        return self._c1
    
    @property
    def c2(self):
        return self._c2
    
    def forward(self, x, u):
        r'''
        Parameters
        ----------
        x : Tensor
            The state of LTI system
        u : Tensor
            The input of LTI system

        Returns
        -------
        z : Tensor
            Derivative of x in discrete case, state-transition
        y : Tensor
            The output of LTI system, observation
            
        Every linear time-invariant lumped system can be described by a set of equations of the form
        which is called the state-space equation.
        
        .. math::
            \begin{align*}
                z_{i} = A_{i} \times x_{i} + B_{i} \times u_{i} + c_1
                y_{i} = C_{i} \times x_{i} + D_{i} \times u_{i} + c_2
            \end{align*}
            
        where :math:`\mathbf{z}` is actually :math:`\mathbf{\dot{x}}`, the differential form of :math:`\mathbf{x}`
        
        Let the input be matrix :math:`\mathbf{A}`, :math:`\mathbf{B}`, :math:`\mathbf{C}`, :math:`\mathbf{D}`, :math:`\mathbf{x}`, :math:`\mathbf{u}`.
        :math:`\mathbf{x}_i` represents each individual matrix in the batch. 
        
        Note:
            -x, u could be single input or multiple inputs

            -A, B, C, D can only be two-dimensional matrices or the batch
             In the batch case, their dimensions must be the same as those of u, x 
             A, B, C, D and u, x are multiplied separately for each channel.
             
            -For a System with p inputs, q outputs, and n state variables,
             A, B, C, D are n*n n*p q*n and q*p constant matrices.
             
            -Note that variables are entered as row vectors.

        Example:
            >>> A = torch.randn((3,3))
                B = torch.randn((3,2))
                C = torch.randn((3,3))
                D = torch.randn((3,2))
                c1 = torch.randn((2,1,3))
                c2 = torch.randn((2,1,3))
                x = torch.randn((2,1,3))
                u = torch.randn((2,1,2))
            >>> A
            tensor([[ 0.3925, -0.1799, -0.0653],
                    [-0.6016,  1.9318,  1.1651],
                    [-0.3182,  1.4565,  1.0184]]) 
                B
            tensor([[-0.4794, -1.7299],
                    [-1.1820, -0.0606],
                    [-1.2021, -0.5444]]) 
                C
            tensor([[-0.1721,  1.6730, -0.6955],
                    [-0.4956,  1.3174,  0.3740],
                    [-0.0835,  0.3706, -1.9351]])
                D
            tensor([[ 1.9300e-01, -1.3445e+00],
                    [ 2.6992e-01, -9.1387e-01],
                    [-6.3274e-04,  5.1283e-01]]) 
                c1
            tensor([[[-0.8519, -0.6737, -0.3359]],
                    [[ 0.5543, -0.1456,  1.4389]]]) 
                c2
            tensor([[[-0.7543, -0.6047, -0.6620]],
                    [[ 0.6252,  2.6831, -3.1711]]]) 
                x
            tensor([[[ 1.0022, -0.1371,  1.0773]],
                    [[ 0.7227,  0.7777,  1.0332]]]) 
                u
            tensor([[[1.7736, 0.7472]],
                    [[0.4841, 0.9187]]])
            >>> lti = LTI(A, B, C, D, c1, c2)
            tensor([[[-1.7951, -1.7544, -1.9603]],
                    [[-1.7451,  1.6436,  0.8730]]]), 
            tensor([[[-1.8134, -0.4785, -1.8370]],
                    [[-0.6836,  0.3439, -1.3006]]]))
    
        Note:
            In this general example, all variables are in the batch. User definable as appropriate.
            
        '''

        if self.A.ndim >= 3:
            assert self.A.ndim == x.ndim == u.ndim,  "Invalid System Matrices dimensions"
        else:
            assert self.A.ndim == 2,  "Invalid System Matrices dimensions"

        z = x.matmul(self.A.mT) + u.matmul(self.B.mT) + self.c1
        y = x.matmul(self.C.mT) + u.matmul(self.D.mT) + self.c2

        return z, y