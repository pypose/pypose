import torch
import functools
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal



class MPPI():
    r'''
    Model Predictive Path Integral Control (MPPI).

    Args:
        dynamics (:obj:`callable`): Function(state, action) -> next_state. The dynamics
            model of the system.
        running_cost (:obj:`callable`): Function to compute the cost for a given state and
            action.
        nx (:obj:`int`): Dimension of the state space.
        noise_sigma (:obj:`Tensor`): Covariance matrix for control noise.
        num_samples (:obj:`int`): Number of trajectories to sample.
        horizon (:obj:`int`): Length of each trajectory.
        device (:obj:`str`): PyTorch device for computation.
        noise_mu (:obj:`Tensor`, optional): Mean of the control noise distribution.
            Defaults to a tensor of zeros.
        lambda_ (:obj:`float`, optional): Temperature parameter that affects the balance
            between cost and exploration. Higher values favor exploration.
        u_min (:obj:`Tensor`, optional): Lower bounds for control actions. Defaults to
            negative infinity.
        u_max (:obj:`Tensor`, optional): Upper bounds for control actions. Defaults to
            positive infinity.
        noise_abs_cost (:obj:`float`, optional): Absolute cost of the noise, aiding in
            control noise evaluation. Defaults to zero.

    **Model Predictive Path Integral Control** is an advanced form of Nonlinear Model
    Predictive Control (NMPC) that efficiently solves finite-horizon optimal control
    tasks under nonlinear dynamics and complex cost functions. It leverages the
    parallel computing power of GPUs to sample multiple trajectories in real-time,
    optimizing control actions based on a weighted average of the sampled costs.

    The operation of MPPI can be described with the following optimization problem,
    aiming to minimize the expected value of the cost function over a prediction
    horizon :math:`N`:

    .. math::
        \begin{align*}
            \min_{\mathbf{U}} J (\mathbf{U}) &= \mathbb{E} \left[ \sum_{k=0}^{N-1}
                \left(q_k(\mathbf{x}_k, \mathbf{u}_k) +
                \frac{1}{2}\mathbf{u}_k^\top\mathbf{R}\mathbf{u}_k\right) +
                \phi(\mathbf{x}_N)\right] \\
            \text{subject to} \quad
            &\mathbf{x}_{k+1} = F(\mathbf{x}_k, \mathbf{u}_k + \bm{\epsilon}_k) \\
            &\mathbf{X}(0) = \mathbf{x}_0, \quad \bm{\epsilon}_k \sim
                \mathcal{N} (0, \bm{\Sigma}_{\epsilon})
        \end{align*}

    Here, :math:`J` is the objective function, :math:`\mathbb{E}` denotes the expected
    value, and :math:`\epsilon` represents the control noise modeled by the
    Ornstein-Uhlenbeck process. MPPI samples :math:`M` trajectories, computes the cost
    for each, and calculates the optimal control as a weighted average of these costs:

    .. math::
        \begin{equation*}
            \mathbf{U} = \sum_{j = 1}^M w_j\mathbf{U}^{(j)}/\sum_{j = 1}^{M} w_j
        \end{equation*}

    Weights :math:`w_j` for each sample are computed as:

    .. math::
        \begin{equation*}
            w_j = \exp\left(-\frac{1}{\lambda}(S_j - \beta)\right)
        \end{equation*}

    where :math:`\beta = \min_{j=1,\cdots, M} S_j`, and :math:`\lambda` influences
    the selectiveness of the weighted average.

    Note:
        MPPI is based on the algorithm from:

        - Williams, G., et al., 'Information Theoretic MPC for Model-Based
          Reinforcement Learning', 2017.
        - Implementations adapted from GitHub repositories for pytorch_mppi
          and mppi_pendulum.


    Example:
        >>> import torch
        >>> import pypose as pp
        >>> class Simple2DNav(pp.module.System):
        ...     def __init__(self,dt,length=1.0):
        ...         super().__init__()
        ...         self._tau = dt
        ...         self._length=length
        ...
        ...     def state_transition(self, state, input, t=None):
        ...         x, y, theta = state.moveaxis(-1, 0)
        ...         v, omega = input.squeeze().moveaxis(-1, 0)
        ...         xDot = v * torch.cos(theta)
        ...         yDot = v * torch.sin(theta)
        ...         thetaDot = omega
        ...         _dstate = torch.stack((xDot, yDot, thetaDot), dim=-1)
        ...         return (state.squeeze() + torch.mul(_dstate, self._tau)).unsqueeze(0)
        ...
        ...     def observation(self, state, input, t=None):
        ...         return state
        ...
        >>> torch.manual_seed(0)
        >>> x0 = torch.tensor([0., 0., 0.], requires_grad=False)
        >>> dt = 0.1
        >>> cost_fn = lambda x, u, t: (x[..., 0] - 10)**2 + (x[..., 1] - 10)**2 \
        >>> + (u[..., 0])**2
        >>> mppi = MPPI(
        >>>     dynamics=Simple2DNav(dt),
        >>>     running_cost=cost_fn,
        >>>     nx=3,
        >>>     noise_sigma=torch.eye(2) * 1,
        >>>     num_samples=100,
        >>>     horizon=5,
        >>>     lambda_=0.01
        >>>     )
        >>> u, xn= pp.mppi.forward(x0)
        >>> print('u = ', u)
        >>> print('x = ', xn)
        u =  tensor([[-1.0977,  0.6999],
                        [ 0.4890, -0.6172],
                        [ 1.3908, -0.6498],
                        [-0.1326, -0.2450],
                        [ 0.6668, -0.9944]])
        x =  tensor([[ 0.0000e+00,  0.0000e+00,  0.0000e+00],
                        [-1.0977e-01,  0.0000e+00,  6.9991e-02],
                        [-6.0990e-02,  3.4199e-03,  8.2744e-03],
                        [ 7.8084e-02,  4.5706e-03, -5.6704e-02],
                        [ 6.4848e-02,  5.3220e-03, -8.1209e-02],
                        [ 1.3131e-01, -8.6915e-05, -1.8065e-01]])
    '''

    def __init__(self,
                 dynamics,
                 running_cost,
                 nx,
                 noise_sigma,
                 num_samples=100,
                 horizon=15.0,
                 device="cpu",
                 lambda_=1.,
                 noise_mu=None,
                 u_min=None,
                 u_max=None,
                 noise_abs_cost=False,
                 batch_size=1
                 ):
        self.d = device
        self.dtype = noise_sigma.dtype
        self.K = num_samples  # N_SAMPLES
        self.T = horizon  # TIMESTEPS
        # dimensions of state and control
        self.nx = nx
        self.nu = 1 if len(noise_sigma.shape) == 0 else noise_sigma.shape[0]
        self.lambda_ = lambda_
        if noise_mu is None:
            noise_mu = torch.zeros(self.nu, dtype=self.dtype)
        # handle 1D edge case
        if self.nu == 1:
            noise_mu = noise_mu.view(-1)
            noise_sigma = noise_sigma.view(-1, 1)

        self.u_max = u_max
        self.u_min = u_min
        if self.u_max is not None and self.u_min is None:
            self.u_max = torch.tensor(self.u_max) \
                if not torch.is_tensor(self.u_max) else self.u_max
            self.u_min = -self.u_max
        if self.u_min is not None and self.u_max is None:
            self.u_min = torch.tensor(self.u_min) \
                if not torch.is_tensor(self.u_min) else self.u_min
            self.u_max = -self.u_min
        if self.u_min is not None:
            self.u_min = self.u_min.to(device=self.d)
            self.u_max = self.u_max.to(device=self.d)
        self.noise_mu = noise_mu.to(self.d)
        self.noise_sigma = noise_sigma.to(self.d)
        self.noise_sigma_inv = torch.inverse(self.noise_sigma)
        self.noise_dist = MultivariateNormal(self.noise_mu, \
            covariance_matrix=self.noise_sigma)
        self.system = dynamics.state_transition
        self.running_cost = running_cost
        self.noise_abs_cost = noise_abs_cost
        self.state = None
        # sampled results from last command
        self.cost_total = None
        self.cost_total_non_zero = None
        self.omega = None
        self.states = None
        self.actions = None
        self.batch_size = batch_size



    def forward(self, state, u_init=None):
        r'''
            Executes the MPPI control loop for a given initial state and optional initial control input.

            Args:
                state (:obj:`Tensor`): The initial state of the system.
                u_init (:obj:`Tensor`, optional): Initial guess for the control inputs. Defaults to None.

            Returns:
                :obj:`Tensor`: The optimal control sequence over the time horizon.
                :obj:`Tensor`: The sequence of states resulting from applying the optimal control sequence.


            The MPPI controller generates a base control sequence and then samples multiple perturbations around this sequence using the specified noise distribution. Constraints on control inputs are enforced, and the costs associated with each trajectory are evaluated to compute a weighted average of the perturbed controls, forming the final optimal control sequence.
        '''
        state = torch.tensor(state, dtype=self.dtype, device=self.d) \
            if not torch.is_tensor(state) else state.to(dtype=self.dtype, device=self.d)
        u_init = torch.zeros_like(self.noise_mu, device=self.d) \
            if u_init is None else u_init.to(self.d)


        U_base = self.noise_dist.sample((self.T,))
        U_base = torch.roll(U_base, -1, dims=0)
        U_base[-1] = u_init

        noise = self.noise_dist.sample((self.K, self.T))
        perturbed_action = U_base + noise

        if self.u_max is not None:
            for t in range(self.T):
                u = perturbed_action[:, slice(t * self.nu, (t + 1) * self.nu)]
                cu = torch.max(torch.min(u, self.u_max), self.u_min)
                perturbed_action[:, slice(t * self.nu, (t + 1) * self.nu)] = cu

        noise = perturbed_action - U_base
        if self.noise_abs_cost:
            action_cost = self.lambda_ * torch.abs(noise) @ self.noise_sigma_inv
        else:
            action_cost = self.lambda_ * noise @ self.noise_sigma_inv

        K, T, nu = perturbed_action.shape
        assert nu == self.nu
        cost_total_temp = torch.zeros(K, device=self.d, dtype=self.dtype)
        cost_samples = cost_total_temp
        states_temp = []
        actions_temp = []

        # Repeat the initial state to match the batch size
        state_temp = state.repeat(K, 1)

        for t in range(T):
            u = perturbed_action[:, t]
            #import pdb; pdb.set_trace()
            state_temp = self.system(state_temp, u)
            #import pdb; pdb.set_trace()
            c = self.running_cost(state_temp, u, t)
            cost_samples = cost_samples + c
            states_temp.append(state_temp)
            actions_temp.append(u)

        cost_total_temp = cost_total_temp + cost_samples.mean(dim=0)
        cost_total = cost_total_temp

        # Action perturbation cost
        perturbation_cost = torch.sum(U_base * action_cost, dim=(1, 2))
        cost_total = cost_total + perturbation_cost

        beta = torch.min(cost_total)
        cost_total_non_zero = torch.exp(-(1 / self.lambda_) * (cost_total - beta))
        eta = torch.sum(cost_total_non_zero)
        self.omega = (1. / eta) * cost_total_non_zero
        for t in range(self.T):
            U_base[t] = U_base[t] + \
                torch.sum(self.omega.view(-1, 1) * noise[:, t], dim=0)


        action2states = [state]
        # Start with initial state in a list
        for i in range(T):
            cur_state = self.system(action2states[i], U_base[i])
            action2states.append(cur_state.squeeze())

        # Convert list of tensors to a single tensor
        action2states = torch.stack(action2states, dim=0)

        return U_base, action2states
