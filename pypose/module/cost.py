import torch as torch
import torch.nn as nn
import pypose as pp
from torch.autograd.functional import jacobian
from torch.autograd import grad


class Cost(nn.Module):
    r'''
    The base class of a cost function.
    
    todo: change description
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
        theory. First, the linearization can be done for arbitrary point(s), not limit to
        the equilibrium point(s), and therefore the extra constant terms :math:`\mathbf{c}_1`
        and :math:`\mathbf{c}_2` are produced. Second, the linearized equations are represented
        by the full states and inputs: :math:`\mathbf{x}` and :math:`\mathbf{u}`, rather than 
        the perturbation format: :math:`\delta \mathbf{x}` and :math:`\delta \mathbf{u}`
        so that the model is consistent with, e.g., the LTI model and the iterative LQR
        solver. More details go to :meth:`LTI`.

    Example:

        A simple linear time-varying system.  Here we just show an example for advancing one
        time step of the system at a given time step and computing the linearization.
        For generating one trajecotry given a series of inputs and advanced use of linearization,
        see the `examples` folder.

        >>> import math
        >>> import pypose as pp
        >>> import torch
        >>> class Floquet(pp.module.System):
        ...     def __init__(self):
        ...         super(Floquet, self).__init__()
        ...
        ...     def state_transition(self, state, input, t):
        ...         cc = torch.cos(2 * math.pi * t / 100)
        ...         ss = torch.sin(2 * math.pi * t / 100)
        ...         A = torch.tensor([[   1., cc/10],
        ...                           [cc/10,    1.]])
        ...         B = torch.tensor([[ss],
        ...                           [1.]])
        ...         return state.matmul(A) + B.matmul(input)
        ...
        ...     def observation(self, state, input, t):
        ...         return state + t
        ...
        >>> solver = Floquet()
        >>> time_curr = 8 # We start from t = 8, and advance one step to t = 9
        >>> input = torch.sin(2 * math.pi * torch.tensor(time_curr) / 50)
        >>> state_curr = torch.tensor([1., 1.])
        >>> solver.reset(t = time_curr)
        >>> state_next, obser_curr = solver(state_curr, input)
        ... 
        >>> solver.set_refpoint()
        >>> print(state_next)
        >>> print(obser_curr)
        >>> print(solver.A)
        >>> print(solver.B)
        tensor([1.4944, 1.9320])
        tensor([9., 9.])
        tensor([[1.0000, 0.0844],
                [0.0844, 1.0000]])
        tensor([[0.5358],
                [1.0000]])

    Note:
        More practical examples can be found at `examples/module/dynamics
        <https://github.com/pypose/pypose/tree/main/examples/module/dynamics>`_.
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
                           - self._ref_state.matmul(self.cxx.mT).matmul(self._ref_state) \
                           - 0.5 * self._ref_state.matmul(self.cux.mT).matmul(self._ref_input) \
                           - 0.5 * self._ref_input.matmul(self.cxu.mT).matmul(self._ref_state) \
                           - self._ref_input.matmul(self.cuu.mT).matmul(self._ref_input)


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