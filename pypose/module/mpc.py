import pypose as pp
import torch.nn as nn


class MPC(nn.Module):
    r'''
    Model Predictive Control (MPC) based on iterative LQR.

    Args:
        system (:obj:`instance`): The system to be soved by MPC.
        T (:obj:`int`): Time steps of system.
        steps (:obj:`int`): Total number of iterations for iterative LQR.
        eps (:obj:`int`): Epsilon as the tolerance.

    MPC requires that at each time step we solve the optimization problem:

    .. math::
        \begin{align*}
          \mathop{\arg\min}\limits_{\mathbf{x}_{1:T}, \mathbf{u}_{1:T}} \sum\limits_t
          \mathbf{c}_t (\mathbf{x}_t, \mathbf{u}_t) \\
          \mathrm{s.t.} \quad \mathbf{x}_{t+1} &= \mathbf{f}(\mathbf{x}_t, \mathbf{u}_t), \\
          \mathbf{x}_1 &= \mathbf{x}_{\text{init}} \\
        \end{align*}

    where :math:`\mathbf{c}` is the cost function; :math:`\mathbf{x}`, :math:`\mathbf{u}`
    are the state and input of the linear system; :math:`\mathbf{f}` is the dynamics model;
    :math:`\mathbf{x}_{\text{init}}` is the initial state of the system.

    We use the iLQR solver to solve the optimization problem, and consider the dynamics as
    constraints. The analytical derivative can be computed using one additional pass of iLQR.


    Note:
        For the lqr solver, please refer to :meth:`LQR`. For linear systems, we only
        require a single iLQR solve.


    Note:
        The implementation of MPC is based on Eq. (10)~(13) of this paper:

        * Amos, Brandon, et al, `Differentiable mpc for end-to-end planning and control
          <https://proceedings.neurips.cc/paper/2018/hash/ba6d843eb4251a4526ce65d1807a9309-Abstract.html>`_,
          Advances in neural information processing systems 31, 2018.

    Example:

    '''
    def __init__(self, system, T, step=10, eps=1e-4):
        super().__init__()
        self.system = system
        self.T = T
        self.step = step
        self.eps = eps

    def forward(self, Q, p, x_init, time, current_u=None):
        r'''
        Performs MPC for the discrete system.

        Args:
            Q (:obj:`Tensor`): The weight matrix of the quadratic term.
            p (:obj:`Tensor`): The weight vector of the first-order term.
            x_init (:obj:`Tensor`): The initial state of the system.
            time (:obj:`Tensor`, optinal): The reference time step of the dynamical
                system.
            current_u (:obj:`Tensor`, optinal): The current inputs of the system along a
                trajectory. Default: ``None``.

        Returns:
            List of :obj:`Tensor`: A list of tensors including the solved state sequence
            :math:`\mathbf{x}`, the solved input sequence :math:`\mathbf{u}`, and the associated
            quadratic costs :math:`\mathbf{c}` over the time horizon.
        '''
        best = None
        u = current_u

        for i in range(self.step):

            lqr = pp.module.LQR(self.system, Q, p, self.T)
            x, u, cost = lqr(x_init, time, u)
            assert x.ndim == u.ndim == 3

            if best is None:
                best = {
                    'u': u,
                    'cost': cost,}
            else:
                if cost <= best['cost']+ self.eps:
                    best['u']= u
                    best['cost'] = cost

        if self.step > 1:
            current_u = best['u']

        _lqr = pp.module.LQR(self.system, Q, p, self.T)
        x, u, cost= _lqr(x_init, time, current_u)

        return x, u, cost
