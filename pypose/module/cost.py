import torch as torch
import torch.nn as nn
import pypose as pp
from torch.autograd.functional import jacobian
from torch.autograd import grad


class Cost(nn.Module):
    r'''
    The base class of a cost function.
    
    todo: change description
    The cost function :math:`\mathrm{cost}` is given by:

    .. math::
        \begin{aligned}
            \mathrm{cost} = \mathrm{func}(\mathbf{x}_k, \mathbf{u}_k)
        \end{aligned}

    where :math:`k`, :math:`\mathbf{x}`, :math:`\mathbf{u}` are the state(s), input(s), respectively.

    Note:
        To use the class, users need to inherit this class and define methods
        :obj:`cost`, which is automatically called by
        internal :obj:`forward` method.

    Note:

        This class provides automatic **linearlization and quadraticization** at a reference point
        :math:`\chi^*=(\mathbf{x}^*, \mathbf{u}^*)`.
        One can directly call those linearized and quadraticized cost as properties including
        :obj:`cx`, :obj:`cu`, :obj:`cxx`, :obj:`cxu`, :obj:`cux`, :obj:`cuu`, and :obj:`c`, after calling
        a method :obj:`set_refpoint`.

    Example:

        A simple non-quadratic cost example. For advanced usage, see the `examples` folder.

        >>> import torch as torch
        >>> class NonQuadCost(Cost):
        ... def __init__(self):
        ...    super(NonQuadCost, self).__init__()
        ... def cost(self, state, input):
        ...    return torch.sum(state**3) + torch.sum(input**4) \
        ...            + torch.sum(state * input) # did not consider batch here
        >>> state = torch.randn(1, 3)
        >>> input = torch.randn(1, 3)
        >>> nonQuadCost = NonQuadCost()
        >>> cost_value = nonQuadCost(state, input)
        >>> print('cost_value', cost_value)
        >>> # 1st, 2nd order partial derivatives at current state and input
        >>> jacob_state, jacob_input = state, input
        >>> nonQuadCost.set_refpoint(state=jacob_state, input=jacob_input)
        >>> print('cx', nonQuadCost.cx.size(), nonQuadCost.cx, '?==?', 3*state**2 + input)
        >>> print('cu', nonQuadCost.cu.size(), nonQuadCost.cu, '?==?', 4*input**3 + state)
        >>> print('cxx', nonQuadCost.cxx.size(), nonQuadCost.cxx, '?==?', 6*torch.diag(state.squeeze(0)))
        >>> print('cxu', nonQuadCost.cxu.size(), nonQuadCost.cxu, '?==?', torch.eye(state.size(-1)) )
        >>> print('cux', nonQuadCost.cux.size(), nonQuadCost.cux, '?==?', torch.eye(state.size(-1)))
        >>> print('cuu', nonQuadCost.cuu.size(), nonQuadCost.cuu, '?==?', 12*torch.diag((input**2).squeeze(0)))
        >>> print('c', nonQuadCost.c.size())
        >>> cx, cu, cxx, cxu, cux, cuu, c = nonQuadCost.cx, nonQuadCost.cu, nonQuadCost.cxx, nonQuadCost.cxu, nonQuadCost.cux, nonQuadCost.cuu, nonQuadCost.c

    Note:
        More practical examples can be found at `examples/module/cost
        <https://github.com/pypose/pypose/tree/main/examples/module/cost>`_.
    '''

    def __init__(self):
        super().__init__()
        self.jacargs = {'vectorize':True, 'strategy':'reverse-mode'}
        # self.register_buffer('_t',torch.zeros(1))
        # self.register_forward_hook(self.forward_hook)

    # def forward_hook(self, module, inputs, outputs):
    #     r'''
    #     Automatically advances the time step.
    #     '''
    #     self._t.add_(1)

    def forward(self, state, input):
        r'''
        Defines the computation performed at every call that advances the system by one time step.

        Note:
            The :obj:`forward` method implicitly increments the time step via :obj:`forward_hook`.
            :obj:`state_transition` and :obj:`observation` still accept time for the flexiblity
            such as time-varying system. One can directly access the current system time via the
            property :obj:`systime`.
        '''
        self.state, self.input = torch.atleast_1d(state), torch.atleast_1d(input)
        return self.cost(self.state, self.input)

    def cost(self, state, input, t=None):
        r'''
        Args:
            state (:obj:`Tensor`): The state of the dynamical system
            input (:obj:`Tensor`): The input to the dynamical system
            t (:obj:`Tensor`): The time step of the dynamical system.  Default: ``None``.

        Returns:
            Tensor: The state of the system at next time step

        Note:
            The users need to define this method and can access the current time via the property
            :obj:`systime`.
        '''
        raise NotImplementedError("The users need to define their own state transition method")

    # def reset(self, t=0):
    #     self._t.fill_(t)

    def set_refpoint(self, state=None, input=None):
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
            None

        Warning:
            For nonlinear systems, the users have to call this function before getting the
            linearized system.
        '''
        self._ref_state = torch.tensor(self.state) if state is None else torch.atleast_1d(state)
        self._ref_input = torch.tensor(self.input) if input is None else torch.atleast_1d(input)
        self._ref_c = self.cost(self._ref_state, self._ref_input)

    # @property
    # def systime(self):
    #     r'''
    #         System time, automatically advanced by :obj:`forward_hook`.
    #     '''
    #     return self._t

    @property
    def cx(self):
        r'''
        Quadratic/quadraticized cost linear term on state
        
        todo: change this
        .. math::
            \mathbf{B} = \left. \frac{\partial \mathbf{f}}{\partial \mathbf{u}} \right|_{\chi^*}
        '''
        func = lambda x: self.cost(x, self._ref_input)
        return jacobian(func, self._ref_state, **self.jacargs)

    @property
    def cu(self):
        r'''
        Quadratic/quadraticized cost linear term on input
        
        todo: change this
        .. math::
            \mathbf{B} = \left. \frac{\partial \mathbf{f}}{\partial \mathbf{u}} \right|_{\chi^*}
        '''
        func = lambda x: self.cost(self._ref_state, x)
        return jacobian(func, self._ref_input, **self.jacargs)    


    @property
    def cxx(self):
        r'''
        Quadratic/quadraticized cost quadratic term on state
        
        todo: change this
        .. math::
            \mathbf{B} = \left. \frac{\partial \mathbf{f}}{\partial \mathbf{u}} \right|_{\chi^*}
        '''
        # https://discuss.pytorch.org/t/second-order-derivatives-of-loss-function/71797/4 need _ref to be requires_grad = True?
        # first_derivative = grad(self.cost(self._ref_state, self._ref_input), self._ref_state, create_graph=True)[0]
        # print('first derivative', first_derivative.size())
        # second_derivative = grad(first_derivative, self._ref_state)[0]
        # print('second derivative', second_derivative)

        # https://pytorch.org/docs/stable/_modules/torch/autograd/functional.html#hessian

        # def jac_func(x): # create a functional here
        #     func = lambda x: self.cost(x, self._ref_input)
        #     jac = jacobian(func, x, create_graph=True)
        #     # print('jac', jac.size())
        #     return jac

        # equivalent simpler form
        func = lambda x: self.cost(x, self._ref_input)
        jac_func = lambda x: jacobian(func, x, create_graph=True) 
        return jacobian(jac_func, self._ref_state, **self.jacargs).squeeze()

    @property
    def cxu(self):
        def jac_func(u):
            func = lambda x: self.cost(x, u)
            jac = jacobian(func, self._ref_state, create_graph=True) # substitute x here
            return jac
        return jacobian(jac_func, self._ref_input,  **self.jacargs).squeeze()
    
    @property
    def cux(self):
        def jac_func(x):
            func = lambda u: self.cost(x, u)
            jac = jacobian(func, self._ref_input, create_graph=True)
            return jac
        return jacobian(jac_func, self._ref_state,  **self.jacargs).squeeze()
 
    @property
    def cuu(self):
        r'''
        Quadratic/quadraticized cost quadratic term on input
        
        todo: change this
        .. math::
            \mathbf{B} = \left. \frac{\partial \mathbf{f}}{\partial \mathbf{u}} \right|_{\chi^*}
        '''
        def jac_func(u):
            func = lambda u: self.cost(self._ref_state, u)
            jac = jacobian(func, u, create_graph=True)
            return jac
        return jacobian(jac_func, self._ref_input, **self.jacargs).squeeze()

    @property
    def c(self):
        r'''
        Constant term generated by cost.

        .. math::
            \mathbf{c}_1 = \mathbf{f}(\mathbf{x}^*, \mathbf{u}^*, t^*)
                           - \mathbf{A}\mathbf{x}^* - \mathbf{B}\mathbf{u}^*
        '''
        # Potential performance loss here - self.A and self.B involves jacobian eval
        return self._ref_c - self._ref_state.matmul(self.cx.mT) - self._ref_input.matmul(self.cu.mT) \
                           - self._ref_state.matmul(self.cxx).matmul(self._ref_state.mT) \
                           - 0.5 * self._ref_state.matmul(self.cxu).matmul(self._ref_input.mT) \
                           - 0.5 * self._ref_input.matmul(self.cux).matmul(self._ref_state.mT) \
                           - self._ref_input.matmul(self.cuu).matmul(self._ref_input.mT)


class QuadCost(Cost):
    r'''
    Quadratic cost.
    
    todo: change
    Args:
        A (:obj:`Tensor`): The state matrix of LTI system.
        B (:obj:`Tensor`): The input matrix of LTI system.
        C (:obj:`Tensor`): The output matrix of LTI system.
        D (:obj:`Tensor`): The observation matrix of LTI system,
        c1 (:obj:`Tensor`): The constant input of LTI system,
        c2 (:obj:`Tensor`): The constant output of LTI system.

    A linear time-invariant lumped system can be described by state-space equation of the form:

    .. math::
        \begin{align*}
            \mathbf{z} = \mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{u} + \mathbf{c}_1 \\
            \mathbf{y} = \mathbf{C}\mathbf{x} + \mathbf{D}\mathbf{u} + \mathbf{c}_2 \\
        \end{align*}

    where :math:`\mathbf{x}` and :math:`\mathbf{u}` are state and input of the current
    timestamp of LTI system.

    Note:
        The variables including state and input are row vectors, which is the last dimension of
        a Tensor. :obj:`A`, :obj:`B`, :obj:`C`, :obj:`D`, :obj:`x`, :obj:`u` could be a single
        matrix or batched matrices. In the batch case, their dimensions must be consistent so that
        they can be multiplied for each channel.

    Example:
        >>> A = torch.randn(2, 3, 3)
        >>> B = torch.randn(2, 3, 2)
        >>> C = torch.randn(2, 3, 3)
        >>> D = torch.randn(2, 3, 2)
        >>> c1 = torch.randn(2, 1, 3)
        >>> c2 = torch.randn(2, 1, 3)
        >>> state = torch.randn(2, 1, 3)
        >>> input = torch.randn(2, 1, 2)
        >>> lti = pp.module.LTI(A, B, C, D, c1, c2)
        >>> lti(state, input)
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
    
    def __init__(self, cx, cu, cxx, cxu, cux, cuu, c=None):
        super(QuadCost, self).__init__()
        assert cxx.ndim in (2, 3), "Invalid cost state Matrices dimensions" # todo ?
        assert cuu.ndim in (2, 3), "Invalid cost input Matrices dimensions"
        assert cx.ndim == cu.ndim == cxx.ndim == cxu.ndim == cux.ndim == cuu.ndim, "Invalid System Matrices dimensions"
        self.cx, self.cu, self.cxx, self.cxu, self.cux, self.cuu = cx, cu, cxx, cxu, cux, cuu
        self.c = c
    
    def forward(self, state, input):
        r'''
        Perform one step advance for the quadratic cost.

        '''
        if self.cx.ndim >= 3:
            assert self.cx.ndim == state.ndim == input.ndim,  "Invalid System Matrices dimensions"

        return super(QuadCost, self).forward(state, input)

    def cost(self, state, input):
        r'''
        Perform one step of LTI state transition.

        .. math::
            \mathbf{z} = \mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{u} + \mathbf{c}_1 \\

        '''
        # print('checkpoint', state.matmul(self.cxx.mT).size() )
        # print(state.matmul(self.cx.mT).size())
        # exit()
        return state.matmul(self.cx.mT) + input.matmul(self.cu.mT) \
                        + state.matmul(self.cxx.mT).matmul(state.mT) \
                        + 0.5 * state.matmul(self.cxu.mT).matmul(input.mT) \
                        + 0.5 * input.matmul(self.cux.mT).matmul(state.mT) \
                        + input.matmul(self.cuu.mT).matmul(input.mT) \
                        + self.c

    @property
    def cx(self):
        r'''
        System state matrix :obj:`cx`
        '''
        return self._cx

    @cx.setter
    def cx(self, cx):
        self._cx = cx

    @property
    def cu(self):
        r'''
        System input matrix :obj:`cu`
        '''
        return self._cu

    @cu.setter
    def cu(self, cu):
        self._cu = cu

    @property
    def cxx(self):
        r'''
        System output matrix :obj:`cxx`
        '''
        return self._cxx

    @cxx.setter
    def cxx(self, cxx):
        self._cxx = cxx

    @property
    def cxu(self):
        r'''
        System observation matrix :obj:`cxu`
        '''
        return self._cxu

    @cxu.setter
    def cxu(self, cxu):
        self._cxu = cxu

    @property
    def cux(self):
        r'''
        System observation matrix :obj:`cux`
        '''
        return self._cux

    @cux.setter
    def cux(self, cux):
        self._cux = cux

    @property
    def cuu(self):
        r'''
        System observation matrix :obj:`cuu`
        '''
        return self._cuu

    @cuu.setter
    def cuu(self, cuu):
        self._cuu = cuu

    @property
    def c(self):
        r'''
        Constant input :obj:`c`
        '''
        return self._c

    @c.setter
    def c(self, c):
        self._c = c

if __name__ == "__main__":
        cx = torch.randn(2, 1, 3)
        cu = torch.randn(2, 1, 2)
        cxx = torch.randn(2, 3, 3)
        cxu = torch.randn(2, 2, 3)
        cux = torch.transpose(cxu, 1,2)
        cuu = torch.randn(2, 2, 2)
        c = torch.randn(2, 1, 1)
        state = torch.randn(2, 1, 3)
        input = torch.randn(2, 1, 2)
        quadcost = pp.module.QuadCost(cx,cu,cxx,cxu,cux,cuu,c)
        print(quadcost(state, input))
        print(quadcost.cx)