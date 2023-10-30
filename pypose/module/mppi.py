import torch
import functools
#import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal

def _ensure_non_zero(cost, beta, factor):
    return torch.exp(-factor * (cost - beta))


def is_tensor_like(x):
    return torch.is_tensor(x) #or type(x) is np.ndarray


def squeeze_n(v, n_squeeze):
    for _ in range(n_squeeze):
        v = v.squeeze(0)
    return v


# from arm_pytorch_utilities, standalone since that package is not on pypi yet
def handle_batch_input(n):
    def _handle_batch_input(func):
        """For func that expect 2D input, handle input that have more than 2 dimensions by
        flattening them temporarily"""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # assume inputs that are tensor-like have compatible shapes and is represented
            # by the first argument
            batch_dims = []
            for arg in args:
                if is_tensor_like(arg):
                    if len(arg.shape) > n:
                        # last dimension is type dependent; all previous ones are batches
                        batch_dims = arg.shape[:-(n - 1)]
                        break
                    elif len(arg.shape) < n:
                        n_batch_dims_to_add = n - len(arg.shape)
                        batch_ones_to_add = [1] * n_batch_dims_to_add
                        args = [v.view(*batch_ones_to_add, *v.shape) if is_tensor_like(v)\
                            else v for v in args]
                        ret = func(*args, **kwargs)
                        if isinstance(ret, tuple):
                            ret = [squeeze_n(v, n_batch_dims_to_add) if is_tensor_like(v)\
                                else v for v in ret]
                            return ret
                        else:
                            if is_tensor_like(ret):
                                return squeeze_n(ret, n_batch_dims_to_add)
                            else:
                                return ret
            # no batches; just return normally
            if not batch_dims:
                return func(*args, **kwargs)
            # reduce all batch dimensions down to the first one
            args = [v.view(-1, *v.shape[-(n - 1):]) if (is_tensor_like(v) and \
                len(v.shape) > 2) else v for v in args]
            ret = func(*args, **kwargs)
            # restore original batch dimensions; keep variable dimension (nx)
            if type(ret) is tuple:
                ret = [v if (not is_tensor_like(v) or len(v.shape) == 0) else (
                    v.view(*batch_dims, *v.shape[-(n - 1):]) if len(v.shape) != 2 \
                        else v.view(-1, *v.shape)) for v in ret]
            else:
                if is_tensor_like(ret):
                    if len(ret.shape) == n:
                        ret = ret.view(*batch_dims, *ret.shape[-(n - 1):])
                    else:
                        ret = ret.view(*batch_dims)
            return ret

        return wrapper

    return _handle_batch_input


class MPPI():
    r'''
    Model Predictive Path Integral (MPPI) control.
    This implementation batch samples the trajectories and scales well with the number of
    samples :math:`K`.

    Args:
        dynamics (callable): Function(state, action) -> next_state. The dynamics model
            of the system.
        running_cost (callable): Function to compute the cost associated with a given
            state and action.
        nx (int): Dimension of the state.
        noise_sigma (Tensor): Control noise covariance.
        num_samples (int): Number of trajectories to sample.
        horizon (int): Length of each trajectory.
        device (str): PyTorch device.
        noise_mu (Tensor, optional): Mean for the control noise distribution. Defaults to
            a tensor of zeros.
        lambda_ (float, optional): Temperature parameter that weighs the importance of
            the cost relative to the control noise. Higher values lead to more exploration.
        u_min (Tensor, optional): Minimum boundary for the control action. Defaults to
            negative infinity.
        u_max (Tensor, optional): Maximum boundary for the control action. Defaults to
            positive infinity.
        noise_abs_cost (float, optional): Absolute cost associated with the noise. Helps
            in weighing the magnitude of noise added to the control. Defaults to zero.

    MPPI is a variant of NMPC that repeatedly solves finite-horizon optimal control tasks
    while utilizing nonlinear dynamics and general cost functions, which can be
    nonquadratic and even discontinuous. Specifically, it samples thousands
    of trajectories around some mean control sequence in real-time by taking advantage of
    the parallel computing capabilities of modern Graphic Processing Units(GPUs), then
    produces an optimal trajectory along with its corresponding control sequence by
    calculating the weighted average of the cost of all samples

    Given current state :math:`X`, control signal :math:`U`, and
    predicting horizon :math:`N`.At each timestep, the MPPI controller aims to solve:

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

    where :math:`J` is the objective function, :math:`\mathbb{E}` is the expected value
    , :math:`q_k` is the stage cost, :math:`\phi` is the terminal cost, :math:`F` is
    denotes the system's dynamics which comprises of hexarotor and contact dynamics,
    and :math:`\epsilon` represents a sampled noise or perturbation added to the control
    sequence at time step :math:`k`, which is modeled by Ornstein-Uhlenbeck process. Under
    this definition, the objective is to minimize the expectation of the state and control
    costs with a random state vector while satisfying the dynamics constraints.

    The MPPI algorithm samples :math:`M` trajectories at each time horizon. Let us denote
    :math:`\mathbf{U}' = \begin{bmatrix} \mathbf{u}_0' & \mathbf{u}_1'
    & \cdots & \mathbf{u}_{N-1}'\end{bmatrix}` as the nominal/mean control sequence,
    :math:`\bm{E}^{(j)} = \begin{bmatrix} \bm{\epsilon}_0^{(j)}
    & \bm{\epsilon}_1^{(j)} & \cdots & \bm{\epsilon}_{N-1}^{(j)}\end{bmatrix}`,
    where :math:`\bm{\epsilon}_k^{(j)} \sim  \mathcal{N}(0, \bm{\Sigma}_{\epsilon})`
    as the control disturbance sequence of the :math:`j^{\mathrm{th}}` sampled trajectory,
    and :math:`\mathbf{U}^{(j)} = \begin{bmatrix} \mathbf{u}_0^{(j)} & \mathbf{u}_1^{(j)}
    & \cdots & \mathbf{u}_{N-1}^{(j)}\end{bmatrix}` as the actual control sequence, such
    that :math:`\mathbf{U}^{(j)} = \mathbf{U}' + \bm{E}^{j}`,
    where :math:`j = \begin{bmatrix} 1 & \cdots & M\end{bmatrix}`.
    The cost of the :math:`j^{\mathrm{th}}` sampled trajectory :math:`S_j` can be computed
    based on previous optimization problem.
    Then, the MPPI generates the optimal control sequence as well as the mean sequence for
    the next iteration by a weighted sum,

    .. math::
        \begin{equation*}
            \mathbf{U} = \sum_{j = 1}^M w_j\mathbf{U}^{(j)}/\sum_{j = 1}^{M} w_j
        \end{equation*}

    where the weights :math:`w_j` of the :math:`j^{\mathrm{th}}` sample is computed as:

    .. math::
        \begin{equation*}
            w_j = \exp\left(-\frac{1}{\lambda}(S_j - \beta)\right)
        \end{equation*}


    to optimize the information-theoretical cost, where :math:`\beta =
    \min_{j=1,\cdots, M} S_j`, and :math:`\lambda` determines how selective is
    the weighted average.

    Note:
        The MPPI approach is based on the algorithm from Williams et al., 2017:
        'Information Theoretic MPC for Model-Based Reinforcement Learning'.
        The implementation is adapted from:
        https://github.com/UM-ARM-Lab/pytorch_mppi
        and https://github.com/ferreirafabio/mppi_pendulum.

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
                 noise_abs_cost=False
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
        # bounds
        self.u_min = u_min
        self.u_max = u_max
        # make sure if any of them is specified, both are specified
        if self.u_max is not None and self.u_min is None:
            if not torch.is_tensor(self.u_max):
                self.u_max = torch.tensor(self.u_max)
            self.u_min = -self.u_max
        if self.u_min is not None and self.u_max is None:
            if not torch.is_tensor(self.u_min):
                self.u_min = torch.tensor(self.u_min)
            self.u_max = -self.u_min
        if self.u_min is not None:
            self.u_min = self.u_min.to(device=self.d)
            self.u_max = self.u_max.to(device=self.d)
        self.noise_mu = noise_mu.to(self.d)
        self.noise_sigma = noise_sigma.to(self.d)
        self.noise_sigma_inv = torch.inverse(self.noise_sigma)
        self.noise_dist = MultivariateNormal(self.noise_mu, \
            covariance_matrix=self.noise_sigma)
        self.system = dynamics
        self.running_cost = running_cost
        self.noise_abs_cost = noise_abs_cost
        self.state = None
        # sampled results from last command
        self.cost_total = None
        self.cost_total_non_zero = None
        self.omega = None
        self.states = None
        self.actions = None

    @handle_batch_input(n=2)
    def _dynamics(self, state, u):
        return self.system(state, u) #if self.step_dependency else self.system(state, u)

    @handle_batch_input(n=2)
    def _running_cost(self, state, u, t):
        return self.running_cost(state, u, t)

    def forward(self, state, u_init=None):
        r'''
        Perform MPPI for discrete system

        Args:
            state (:obj:`Tensor`): The initial state of the system.
            u_init (:obj:`Tensor`, optinal): The current inputs of the system along a
                trajectory. Default: ``None``.

        Returns:
            List of :obj:`Tensor`: A list of tensors including the solved input
            sequence :math:`\mathbf{u}` and the solved state sequence :math:`\mathbf{x}`
        '''
        # shift command 1 time step
        if u_init is None:
            u_init = torch.zeros_like(self.noise_mu)
        self.u_init = u_init.to(self.d)
        self.U = self.noise_dist.sample((self.T,))
        self.U = torch.roll(self.U, -1, dims=0)
        self.U[-1] = self.u_init
        if not torch.is_tensor(state):
            state = torch.tensor(state)
        self.state = state.to(dtype=self.dtype, device=self.d)
        cost_total = self._compute_total_cost_batch()
        beta = torch.min(cost_total)
        self.cost_total_non_zero = _ensure_non_zero(cost_total, beta, 1 / self.lambda_)
        eta = torch.sum(self.cost_total_non_zero)
        self.omega = (1. / eta) * self.cost_total_non_zero
        for t in range(self.T):
            self.U[t] = self.U[t] + \
                torch.sum(self.omega.view(-1, 1) * self.noise[:, t], dim=0)
        action = self.U[:self.T]
        action2states=state.unsqueeze(0)
        for i in range(self.T):
            cur_state = self.system(action2states[-1], action[i])[0]
            action2states=torch.cat((action2states,cur_state),dim=0)
        return action, action2states

    def _compute_rollout_costs(self, perturbed_actions):
        K, T, nu = perturbed_actions.shape
        assert nu == self.nu
        cost_total = torch.zeros(K, device=self.d, dtype=self.dtype)
        #cost_samples = cost_total.repeat(self.M, 1)
        cost_samples = cost_total
        # allow propagation of a sample of states (ex. to carry a distribution),
        # or to start with a single state
        if self.state.shape == (K, self.nx):
            state = self.state.clone()
        else:
            state = self.state.view(1, -1).repeat(K, 1).clone()
        states = []
        actions = []
        for t in range(T):
            u = perturbed_actions[:, t]
            state = self._dynamics(state, u)[0]
            c = self._running_cost(state, u, t)
            cost_samples = cost_samples + c
            states.append(state)
            actions.append(u)
        # Actions is K x T x nu
        # States is K x T x nx
        actions = torch.stack(actions, dim=-2)
        states = torch.stack(states, dim=-2)
        cost_total = cost_total + cost_samples.mean(dim=0)
        return cost_total, states, actions

    def _compute_total_cost_batch(self):
        # parallelize sampling across trajectories
        # resample noise each time we take an action
        self.noise = self.noise_dist.sample((self.K, self.T))
        # broadcast own control to noise over samples; now it's K x T x nu
        self.perturbed_action = self.U + self.noise
        # naively bound control
        self.perturbed_action = self._bound_action(self.perturbed_action)
        # bounded noise after bounding (some got cut off,
        # so we don't penalize that in action cost)
        self.noise = self.perturbed_action - self.U
        if self.noise_abs_cost:
            action_cost = self.lambda_ * torch.abs(self.noise) @ self.noise_sigma_inv
            # NOTE: The original paper does
            # self.lambda_ * torch.abs(self.noise) @ self.noise_sigma_inv,
            # but this biases the actions with low noise if all states have the same cost.
            # With abs(noise) we prefer actions close to the nomial trajectory.
        else:# Like original paper
            action_cost = self.lambda_ * self.noise @ self.noise_sigma_inv
        self.cost_total, self.states, self.actions =\
            self._compute_rollout_costs(self.perturbed_action)
        # action perturbation cost
        perturbation_cost = torch.sum(self.U * action_cost, dim=(1, 2))
        self.cost_total = self.cost_total + perturbation_cost
        return self.cost_total

    def _bound_action(self, action):
        if self.u_max is not None:
            for t in range(self.T):
                u = action[:, self._slice_control(t)]
                cu = torch.max(torch.min(u, self.u_max), self.u_min)
                action[:, self._slice_control(t)] = cu
        return action

    def _slice_control(self, t):
        return slice(t * self.nu, (t + 1) * self.nu)
