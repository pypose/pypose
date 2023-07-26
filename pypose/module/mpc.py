import torch
from .lqr import LQR
from torch import nn
from ..utils.stepper import ReduceToBason


class MPC(nn.Module):
    r'''
    Model Predictive Control (MPC) based on iterative LQR.

    Args:
        system (:obj:`instance`): The system to be soved by MPC.
        Q (:obj:`Tensor`): The weight matrix of the quadratic term.
        p (:obj:`Tensor`): The weight vector of the first-order term.
        T (:obj:`int`): Time steps of system.
        stepper (``Planner``, optional): the stepper to stop iterations. If ``None``,
            the ``pypose.utils.ReduceToBason`` with a maximum of 10 steps are used.
            Default: ``None``.

    **Model Predictive Control**, also known as Receding Horizon Control (RHC), uses the
    mathematical model of the system to solve a finite, moving horizon, and
    closed loop optimal control problem. Thus, the MPC scheme is able to utilize the
    information about the current state of the system to predict future states
    and control inputs for the system. At each time stamp, MPC solves the
    optimization problem on a time horizon :math:`T`:

    .. math::
        \begin{align*}
          \mathop{\arg\min}\limits_{\mathbf{x}_{1:T}, \mathbf{u}_{1:T}} \sum\limits_t
          &\mathbf{c}_t (\mathbf{x}_t, \mathbf{u}_t) \\
          \mathrm{s.t.}\quad \mathbf{x}_{t+1} &= \mathbf{f}(\mathbf{x}_t, \mathbf{u}_t),\\
          \mathbf{x}_1 &= \mathbf{x}_{\text{init}} \\
        \end{align*}

    where :math:`\mathbf{c}` is the cost function; :math:`\mathbf{x}`, :math:`\mathbf{u}`
    are the state and input of the linear system; :math:`\mathbf{f}` is the dynamics
    model; :math:`\mathbf{x}_{\text{init}}` is the initial state of the system.

    For the linear system with quadratic cost, MPC is equivalent to solving an LQR problem
    on each horizon as well as considering the dynamics as constraints. For nonlinear
    system, one way to solve the MPC problem is to use iterative LQR, which uses linear
    approximations of the dynamics and quadratic approximations of the cost function to
    iteratively compute a local optimal solution based on the current states and control
    sequences. Then, analytical derivative for the backward can be computed using one
    additional forward pass of iLQR for the learning problem, such as learning the
    parameters of the dynamic system.

    Specifically, the discrete-time non-linear system can be described as:

    .. math::
        \begin{aligned}
            \mathbf{x}_{t+1} &= \mathbf{f}(\mathbf{x}_t, \mathbf{u}_t, t_t) \\
            \mathbf{y}_{t} &= \mathbf{g}(\mathbf{x}_t, \mathbf{u}_t, t_t) \\
        \end{aligned}

    We can do a linear approximation at current point :math:`\chi^*=(\mathbf{x}^*,
    \mathbf{u}^*, t^*)` along a trajectory with small perturbation
    :math:`\chi=(\mathbf{x}^*+\delta\mathbf{x}, \mathbf{u}^* +\delta\mathbf{u}, t^*)`
    near :math:`\chi^*` for both dynamics and cost:

    .. math::
            \begin{aligned}
            \mathbf{f}(\mathbf{x}, \mathbf{u}, t^*) &\approx \mathbf{f}(\mathbf{x}^*,
                \mathbf{u}^*, t^*) + \left.\frac{\partial \mathbf{f}}{\partial \mathbf{x}}
                \right|_{\chi^*} \delta \mathbf{x} + \left. \frac{\partial \mathbf{f}}
                {\partial \mathbf{u}} \right|_{\chi^*} \delta \mathbf{u} \\
            &= \mathbf{f}(\mathbf{x}^*, \mathbf{u}^*, t^*) + \mathbf{A} \delta \mathbf{x}
                + \mathbf{B} \delta \mathbf{u} \\
            \delta \mathbf{x}_{t+1} &= \mathbf{A}_t \delta \mathbf{x}_t + \mathbf{B}_t
                \delta \mathbf{u}_t \\
            &= \mathbf{F}_t \delta \mathbf{\tau}_t \\
            \mathbf{c} \left( \mathbf{\tau}, t^* \right) &\approx
                \mathbf{c} \left(\mathbf{\tau}^*, t^* \right) + \frac{1}{2} \delta
                \mathbf{\tau}^\top\nabla^2_{\mathbf{\tau}}\mathbf{c}\left(\mathbf{\tau}^*,
                t^* \right) \delta \mathbf{\tau} + \nabla_{\mathbf{\tau}}
                \mathbf{c} \left( \mathbf{\tau}^*, t^* \right)^\top \delta \mathbf{\tau}\\
            \bar{\mathbf{c}} \left( \delta \mathbf{\tau} \right) &= \frac{1}{2} \delta
                \mathbf{\tau}_t^\top \bar{\mathbf{Q}}_t \delta \mathbf{\tau}_t +
                \bar{\mathbf{p}}_t^\top \delta \mathbf{\tau}_t \\
            \end{aligned}

    where :math:`\mathbf{F}_t` = :math:`\begin{bmatrix}
    \mathbf{A}_t & \mathbf{B}_t \end{bmatrix}`.

    Then, LQR can be performed on a linear quadractic problem with
    :math:`\delta \mathbf{\tau}_t`, :math:`\mathbf{F}_t`, :math:`\bar{\mathbf{Q}}_t` and
    :math:`\bar{\mathbf{p}}_t`.

    - The backward recursion.

      For :math:`t` = :math:`T` to 1:

        .. math::
            \begin{align*}
                \mathbf{Q}_t &= \bar{\mathbf{Q}}_t + \mathbf{F}_t^\top\mathbf{V}_{t+1}
                                    \mathbf{F}_t \\
                \mathbf{q}_t &= \bar{\mathbf{p}}_t + \mathbf{F}_t^\top\mathbf{v}_{t+1}  \\
                \mathbf{K}_t &= -\mathbf{Q}_{\delta \mathbf{u}_t,\delta \mathbf{u}_t}^{-1}
                                 \mathbf{Q}_{\delta \mathbf{u}_t,\delta \mathbf{x}_t} \\
                \mathbf{k}_t &= -\mathbf{Q}_{\delta \mathbf{u}_t,\delta \mathbf{u}_t}^{-1}
                                    \mathbf{q}_{\delta \mathbf{u}_t}         \\
                \mathbf{V}_t &= \mathbf{Q}_{\delta \mathbf{x}_t, \delta \mathbf{x}_t}
                    + \mathbf{Q}_{\delta \mathbf{x}_t, \delta \mathbf{u}_t}\mathbf{K}_t
                    + \mathbf{K}_t^\top\mathbf{Q}_{\delta\mathbf{u}_t,\delta \mathbf{x}_t}
                    + \mathbf{K}_t^\top\mathbf{Q}_{\delta\mathbf{u}_t,\delta \mathbf{u}_t}
                        \mathbf{K}_t   \\
                \mathbf{v}_t &= \mathbf{q}_{\delta \mathbf{x}_t}
                    + \mathbf{Q}_{\delta \mathbf{x}_t, \delta \mathbf{u}_t}\mathbf{k}_t
                    + \mathbf{K}_t^\top\mathbf{q}_{\delta \mathbf{u}_t}
                    + \mathbf{K}_t^\top\mathbf{Q}_{\delta \mathbf{u}_t,
                        \delta \mathbf{u}_t}\mathbf{k}_t   \\
            \end{align*}

    - The forward recursion.

      For :math:`t` = 1 to :math:`T`:

        .. math::
            \begin{align*}
                \delta \mathbf{u}_t &= \mathbf{K}_t \delta \mathbf{x}_t + \mathbf{k}_t \\
                \mathbf{u}_t &= \delta \mathbf{u}_t + \mathbf{u}_t^* \\
                \mathbf{x}_{t+1} &= \mathbf{f}(\mathbf{x}_t, \mathbf{u}_t) \\
            \end{align*}

    Note:
        For the details of lqr solver and the iterative LQR, please refer to :meth:`LQR`.
        Please note that the linear system with quadratic cost only requires one single
        iteration.

    Note:
        The definition of MPC is cited from this book:

        * George Nikolakopoulos, Sina Sharif Mansouri, Christoforos Kanellakis, `Aerial
          Robotic Workers: Design, Modeling, Control, Vision and Their Applications
          <https://doi.org/10.1016/C2017-0-02260-7>`_, Butterworth-Heinemann, 2023.

        The implementation of MPC is based on Eq. (10)~(13) of this paper:

        * Amos, Brandon, et al, `Differentiable mpc for end-to-end planning and control
          <https://tinyurl.com/3h9tsuyb>`_, Advances in neural information processing
          systems, 31, 2018.

    Example:
        >>> import torch, pypose as pp
        >>> class CartPole(pp.module.NLS):
        ...     def __init__(self, dt, length, cartmass, polemass, gravity):
        ...         super().__init__()
        ...         self.tau = dt
        ...         self.length = length
        ...         self.cartmass = cartmass
        ...         self.polemass = polemass
        ...         self.gravity = gravity
        ...         self.poleml = self.polemass * self.length
        ...         self.totalMass = self.cartmass + self.polemass
        ...
        ...     def state_transition(self, state, input, t=None):
        ...         x, xDot, theta, thetaDot = state.squeeze()
        ...         force = input.squeeze()
        ...         costheta = torch.cos(theta)
        ...         sintheta = torch.sin(theta)
        ...         temp = (force + self.poleml * thetaDot**2 * sintheta) / self.totalMass
        ...         thetaAcc = (self.gravity * sintheta - costheta * temp) /
        ...                     (self.length * (4.0 / 3.0
        ...                            - self.polemass * costheta**2 / self.totalMass))
        ...         xAcc = temp - self.poleml * thetaAcc * costheta / self.totalMass
        ...         _dstate = torch.stack((xDot, xAcc, thetaDot, thetaAcc))
        ...         return (state.squeeze() + torch.mul(_dstate, self.tau)).unsqueeze(0)
        ...
        ...     def observation(self, state, input, t=None):
        ...         return state
        ...
        >>> torch.manual_seed(0)
        >>> dt, len, m_cart, m_pole, g = 0.01, 1.5, 20, 10, 9.81
        >>> n_batch, T = 1, 5
        >>> n_state, n_ctrl = 4, 1
        >>> n_sc = n_state + n_ctrl
        >>> Q = torch.tile(torch.eye(n_state + n_ctrl, device=device), (n_batch, T, 1, 1))
        >>> p = torch.randn(n_batch, T, n_sc)
        >>> time  = torch.arange(0, T, device=device) * dt
        >>> current_u = torch.sin(time).unsqueeze(1).unsqueeze(0)
        >>> x_init = torch.tensor([[0, 0, torch.pi, 0]])
        >>> stepper = pp.utils.ReduceToBason(steps=15, verbose=True)
        >>> cartPoleSolver = CartPole(dt, len, m_cart, m_pole, g).to(device)
        >>> MPC = pp.module.MPC(cartPoleSolver, Q, p, T, stepper=stepper).to(device)
        >>> x, u, cost = MPC(dt, x_init, u_init=current_u)
        >>> print("x = ", x)
        >>> print("u = ", u)
        x =  tensor([[[ 0.0000e+00,  0.0000e+00,  3.1416e+00,  0.0000e+00],
                      [ 0.0000e+00, -3.7711e-04,  3.1416e+00, -1.8856e-04],
                      [-3.7711e-06, -3.0693e-04,  3.1416e+00, -1.5347e-04],
                      [-6.8404e-06, -2.4288e-04,  3.1416e+00, -1.2136e-04],
                      [-9.2692e-06, -2.6634e-04,  3.1416e+00, -1.3293e-04],
                      [-1.1933e-05, -2.2073e-04,  3.1416e+00, -1.0991e-04]]])
        u =  tensor([[[-0.8485],
                      [ 0.1579],
                      [ 0.1440],
                      [-0.0530],
                      [ 0.1023]]])
    '''
    def __init__(self, system, Q, p, T, stepper=None):
        super().__init__()
        self.stepper = ReduceToBason(steps=10) if stepper is None else stepper
        self.stepper.max_steps -= 1 # n-1 loops, 1 loop with gradient
        self.lqr = LQR(system, Q, p, T)

    def forward(self, dt, x_init, u_init=None, u_lower=None, u_upper=None, du=None):
        r'''
        Performs MPC for the discrete system.

        Args:
            dt (:obj:`int`): The interval (:math:`\delta t`) between two time steps.
            x_init (:obj:`Tensor`): The initial state of the system.
            u_init (:obj:`Tensor`, optinal): The current inputs of the system along a
                trajectory. Default: ``None``.
            u_lower (:obj:`Tensor`, optinal): The lower bounds on the controls.
                Default: ``None``.
            u_upper (:obj:`Tensor`, optinal): The upper bounds on the controls.
                Default: ``None``.
            du (:obj:`int`, optinal): The amount each component of the controls
                is allowed to change in each LQR iteration. Default: ``None``.

        Returns:
            List of :obj:`Tensor`: A list of tensors including the solved state sequence
            :math:`\mathbf{x}`, the solved input sequence :math:`\mathbf{u}`, and the
            associated quadratic costs :math:`\mathbf{c}` over the time horizon.
        '''
        x, u = None, u_init
        best = {'x': x, 'u': u, 'cost': None}

        self.stepper.reset()
        with torch.no_grad():
            while self.stepper.continual():
                x, u, cost = self.lqr(x_init, dt, u)
                self.stepper.step(cost)

                if best['cost'] == None or cost < best['cost']:
                    best = {'x': x, 'u': u, 'cost': cost}

        return self.lqr(x_init, dt, u_traj=best['u'])
