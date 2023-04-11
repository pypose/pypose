import pypose as pp
import torch as torch
import torch.nn as nn
from torch.autograd.functional import jacobian

class Cost(nn.Module):
    r'''
    The base class of a cost function.
    
    The cost function :math:`\mathrm{cost}` is given by:

    .. math::
        \begin{aligned}
            \mathrm{cost} = \mathrm{cost}(\mathbf{x}_k, \mathbf{u}_k)
        \end{aligned}

    where :math:`\mathbf{x}_{k}`, :math:`\mathbf{u}_{k}` are the state and input at time instant :math:`k`, respectively.

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
        # self.jacargs = {'vectorize':False, 'strategy':'reverse-mode'}

    def forward(self, state, input):
        r'''
        Defines the computation performed at every call.
        '''
        self.state, self.input = torch.atleast_1d(state), torch.atleast_1d(input)
        return self.cost(self.state, self.input)

    def cost(self, state, input):
        r'''
        Args:
            state (:obj:`Tensor`): The state of the dynamical system
            input (:obj:`Tensor`): The input to the dynamical system

        Returns:
            Tensor: The stage cost of the system
        '''
        raise NotImplementedError("The users need to define their own cost method")

    def set_refpoint(self, state=None, input=None):
        r'''
        Function to set the reference point for linearization.

        Args: 
            state (:obj:`Tensor`): The reference state of the dynamical system. If ``None``,
                the the most recent state is taken. Default: ``None``.
            input (:obj:`Tensor`): The reference input to the dynamical system. If ``None``,
                the the most recent input is taken. Default: ``None``.

        Returns:
            None

        Warning:
            For nonlinear cost, the users have to call this function before getting the
            linearized and quadraticized cost.
        '''
        self._ref_state = torch.tensor(self.state) if state is None else torch.atleast_1d(state)
        self._ref_input = torch.tensor(self.input) if input is None else torch.atleast_1d(input)
        self._ref_c = self.cost(self._ref_state, self._ref_input)

    @property
    def cx(self):
        r'''
        Quadratic/quadraticized cost linear term on state
        
        .. math::
            c_{\mathbf{x}} = \left. \frac{\partial c}{\partial \mathbf{x}} \right|_{\chi^*}
        '''
        func = lambda x: self.cost(x, self._ref_input)
        return excludeBatch(jacobian(func, self._ref_state, **self.jacargs))

    @property
    def cu(self):
        r'''
        Quadratic/quadraticized cost linear term on input
        
        .. math::
            c_{\mathbf{u}} = \left. \frac{\partial c}{\partial \mathbf{u}} \right|_{\chi^*}
        '''
        func = lambda x: self.cost(self._ref_state, x)
        return excludeBatch(jacobian(func, self._ref_input, **self.jacargs))    

    @property
    def cxx(self):
        r'''
        Quadratic/quadraticized cost quadratic term on state
        
        .. math::
            c_\mathbf{xx} = \left. \frac{\partial^{2} c}{\partial \mathbf{x}^{2}} \right|_{\chi^*}
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
        jac_func = lambda x: excludeBatch(jacobian(func, x, create_graph=True))
        return excludeBatch(jacobian(jac_func, self._ref_state, **self.jacargs),type=2)
    
    @property
    def cxu(self):
        r'''
        Quadratic/quadraticized cost quadratic term on state and input
        
        .. math::
            c_\mathbf{xu} = \left. \frac{\partial^{2} c}{\partial \mathbf{x} \partial \mathbf{u}} \right|_{\chi^*}
        '''
        def jac_func(u):
            func = lambda x: self.cost(x, u)
            jac = jacobian(func, self._ref_state, create_graph=True) # substitute x here
            return excludeBatch(jac)
        # func = lambda x,u: self.cost(x, u)
        # jac_func = lambda u: excludeBatch(jacobian(func, x, create_graph=True))        
        return excludeBatch(jacobian(jac_func, self._ref_input,  **self.jacargs), type=2)
    
    @property
    def cux(self):
        r'''
        Quadratic/quadraticized cost quadratic term on input and state
        
        .. math::
            c_\mathbf{ux} = \left. \frac{\partial^{2} c}{\partial \mathbf{u} \partial \mathbf{x}} \right|_{\chi^*}
        '''
        return self.cxu.mT
    
    @property
    def cuu(self):
        r'''
        Quadratic/quadraticized cost quadratic term on input
        
        .. math::
            c_\mathbf{uu} = \left. \frac{\partial^{2} c}{\partial \mathbf{u}^{2}} \right|_{\chi^*}
        '''
        func = lambda u: self.cost(self._ref_state, u)
        jac_func = lambda u: excludeBatch(jacobian(func, u, create_graph=True))
        return excludeBatch(jacobian(jac_func, self._ref_input,  **self.jacargs), type=2)
    
    @property
    def c(self):
        r'''
        Constant term generated by cost.

        .. math::
            c = \mathrm{cost}(\mathbf{x}^*, \mathbf{u}^*)
                           - c_{\mathbf{x}}\mathbf{x}^* - c_{\mathbf{u}}\mathbf{u}^*
                           - \frac{1}{2} \mathbf{x}^{*\top}c_{\mathbf{xx}}\mathbf{x}^*
                           - \frac{1}{2} \mathbf{x}^{*\top}c_{\mathbf{xu}}\mathbf{u}^*
                           - \frac{1}{2} \mathbf{u}^{*\top}c_{\mathbf{ux}}\mathbf{x}^*
                           - \frac{1}{2} \mathbf{u}^{*\top}c_{\mathbf{uu}}\mathbf{u}^*
        '''
        # Potential performance loss here - involves jacobian eval
        return self._ref_c - pp.bdot(self._ref_state, self.cx) - pp.bdot(self._ref_input, self.cu) \
                           - 0.5 * pp.bvmv(self._ref_state, self.cxx, self._ref_state) \
                           - 0.5 * pp.bvmv(self._ref_state, self.cxu, self._ref_input) \
                           - 0.5 * pp.bvmv(self._ref_input, self.cux, self._ref_state) \
                           - 0.5 * pp.bvmv(self._ref_input, self.cuu, self._ref_input)  

def excludeBatch(inp, type=1):
    B = inp.shape[-3:-1]
    if type == 1: # zero dim per sample
        out = torch.zeros(inp.shape[-3:], dtype=inp.dtype, device=inp.device)
        for i in range(B[0]): #todo: compatible with non-batch case
            for j in range(B[1]):
                out[i,j,:] = inp[i,j,i,j,:]
    if type == 2: # vector per sample
        out = torch.zeros(inp.shape[:3]+(inp.shape[-1:]), dtype=inp.dtype, device=inp.device)
        for i in range(B[0]):
            for j in range(B[1]):
                out[i,j,:,:] = inp[i,j,:,i,j,:]
    return out        

class QuadCost(Cost):
    r'''
    Quadratic cost.

    Args:
        Q (:obj:`Tensor`): The quadratic state coefficient matrix of Quadratic cost.
        R (:obj:`Tensor`): The quadratic input coefficient matrix of Quadratic cost.
        S (:obj:`Tensor`): The quadratic cross coefficient matrix of Quadratic cost.
        c (:obj:`Tensor`): The constant term of Quadratic cost.

    A quadratic cost can be described by equation of the form:

    .. math::
        \mathrm{cost} = \frac{1}{2}(\mathbf{x}^{\top}\mathbf{Q}\mathbf{x} 
            +  \mathbf{x}^{\top}\mathbf{S}\mathbf{u} 
            +  \mathbf{u}^{\top}\mathbf{S}^{\top}\mathbf{x} 
            +  \mathbf{u}^{\top}\mathbf{R}\mathbf{u}) + c \\

    where :math:`\mathbf{x}` and :math:`\mathbf{u}` are state and input of the current
    timestamp of system.
    
    Note:
        More practical examples can be found at `examples/module/cost
        <https://github.com/pypose/pypose/tree/main/examples/module/cost>`_.
    '''
    
    def __init__(self, Q, R, S, c=None):
        super(QuadCost, self).__init__()
        assert Q.ndim == R.ndim == S.ndim, "Invalid System Matrices dimensions"
        self.Q, self.R, self.S, self.c = Q, R, S, c
        self.cxx, self.cuu, self.cxu = Q, R, S

    def forward(self, state, input):
        r'''
        Perform forward step for the quadratic cost.

        '''
        return super(QuadCost, self).forward(state, input)

    def cost(self, state, input):
        r'''
        Perform QuadCost computation.

        .. math::
            \mathrm{cost} = \frac{1}{2}(\mathbf{x}^{\top}\mathbf{Q}\mathbf{x} 
                +  \mathbf{x}^{\top}\mathbf{S}\mathbf{u} 
                +  \mathbf{u}^{\top}\mathbf{S}^{\top}\mathbf{x} 
                +  \mathbf{u}^{\top}\mathbf{R}\mathbf{u}) + c \\

        '''
        return (  0.5 * pp.bvmv(state, self.Q, state) \
                + 0.5 * pp.bvmv(state, self.S, input) \
                + 0.5 * pp.bvmv(input, self.S.mT, state) \
                + 0.5 * pp.bvmv(input, self.R, input)
                + self.c)

    # cx, cu cannot be defined via setter

    @property
    def cxx(self):
        r'''
        quadratic term on state :obj:`cxx`
        '''
        return self._cxx

    @cxx.setter
    def cxx(self, cxx):
        self._cxx = cxx

    @property
    def cxu(self):
        r'''
        quadratic cross-term :obj:`cxu`
        '''
        return self._cxu

    @cxu.setter
    def cxu(self, cxu):
        self._cxu = cxu
        self._cux = cxu.mT # set cux simultaneously

    @property
    def cux(self):
        r'''
        quadratic cross-term :obj:`cux`
        '''
        return self._cux

    @property
    def cuu(self):
        r'''
        quadratic term on state :obj:`cuu`
        '''
        return self._cuu

    @cuu.setter
    def cuu(self, cuu):
        self._cuu = cuu

    @property
    def c(self):
        r'''
        Constant term :obj:`c`
        '''
        return self._c

    @c.setter
    def c(self, c):
        self._c = c
