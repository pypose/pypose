import torch as torch
import torch.nn as nn
import pypose as pp
from torch.autograd.functional import jacobian


class _System(nn.Module):
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
        state = self.state_transition(state, input)
        return self.observation(state, input)

    def state_trasition(self):
        pass

    def observation(self):
        pass

    def reset(self,t=0):
        self.t.fill_(0)

    @property
    def A(self):
        if hasattr(self, '_A'):
            return self._A
        else:
            func = lambda x: self.state_trasition(x, self.input)
            return jacobian(func, self.state, **self.jacargs)

    @property
    def B(self):
        if hasattr(self, '_B'):
            return self._B
        else:
            func = lambda x: self.state_trasition(self.state, x)
            return jacobian(func, self.input, **self.jacargs)

    @property
    def C(self):
        if hasattr(self, '_C'):
            return self._C
        else:
            func = lambda x: self.observation(x, self.input)
            return jacobian(func, self.state, **self.jacargs)
 
    @property
    def D(self):
        if hasattr(self, '_D'):
            return self._D
        else:
            func = lambda x: self.observation(self.state, x)
            return jacobian(func, self.input, **self.jacargs)


class LTI(_System):
    r'''
    A sub-class of '_System' to represent the dynamics of discrete-time Linear Time-Invariant (LTI) system.
    
    Args:
        A, B, C, D (:obj:`Tensor`): The coefficient matrix in the state-space equation of LTI system,
        c1 (:obj:`Tensor`): The constant input of the system,
        c2 (:obj:`Tensor`): The constant output of the system.
        
    Note:
        According to the actual physical meaning, the dimensions of A, B, C, D must be the consistent,
        whether in batch or not.
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
    
    def forward(self, state, input):
        r'''
        Args:
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
            :math:`\mathbf{A}`, :math:`\mathbf{B}`, :math:`\mathbf{C}`, :math:`\mathbf{D}`, :math:`\mathbf{x}`, :math:`\mathbf{u}` could be a single input or in a batch.
            In the batch case, their dimensions must be consistent so that they can be multiplied for each channel.
             
            Note that here variables are given as row vectors.

        Example:
            >>> A = torch.randn((2,3,3))
                B = torch.randn((2,3,2))
                C = torch.randn((2,3,3))
                D = torch.randn((2,3,2))
                c1 = torch.randn((2,1,3))
                c2 = torch.randn((2,1,3))
                state = torch.randn((2,1,3))
                input = torch.randn((2,1,2))
            >>> lti = pp.module.LTI(A, B, C, D, c1, c2)
            >>> lti(state,input)
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

        z = self.state_trasition(state, input)
        y = self.observation(state, input)
        return z, y

    def state_trasition(self, state, input):
        state_transition = state.matmul(self._A.mT) + input.matmul(self._B.mT) + self._c1
        return state_transition

    def observation(self, state, input):
        observation = state.matmul(self._C.mT) + input.matmul(self._D.mT) + self._c2
        return observation