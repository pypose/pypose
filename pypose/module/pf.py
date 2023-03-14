import torch
from . import EKF
from ..basics import bvv
import torch.nn.functional as F
from torch.distributions import MultivariateNormal


class PF(EKF):
    r'''
    Performs Batched Particle Filter (PF).

    Args:
        model (:obj:`System`): The system model to be estimated, a subclass of
            :obj:`pypose.module.System`.
        Q (:obj:`Tensor`, optional): The covariance matrices of system transition noise.
            Ignored if provided during each iteration. Default: ``None``
        R (:obj:`Tensor`, optional): The covariance matrices of system observation noise.
            Ignored if provided during each iteration. Default: ``None``
        particles (:obj:`Int`, optional): The number of particle. Default: ``1000``

    A non-linear system can be described as

    .. math::
        \begin{aligned}
            \mathbf{x}_{k+1} &= \mathbf{f}(\mathbf{x}_k, \mathbf{u}_k, t_k) + \mathbf{w}_k,
            \quad \mathbf{w}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{Q})  \\
            \mathbf{y}_{k} &= \mathbf{g}(\mathbf{x}_k, \mathbf{u}_k, t_k) + \mathbf{v}_k,
            \quad \mathbf{v}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{R})
        \end{aligned}

    Particle filter can be described as the following equations, where the subscript
    :math:`\cdot_{k}` is omited for simplicity.

    1. Generate Particles.

        .. math::
            \begin{aligned}
                \mathbf{x} _{k} = \mathbf{p} (\mathbf{x},n*\mathbf{P}_{k},N) \quad k=1,...,N \\
                \mathbf{P} _{k} = \mathbf{P} \quad k=1,...,N
            \end{aligned}

       where :math:`N` is the number of Particles and :math:`\mathbf{p}` is the probability
       density function (PDF), :math:`n` is the dimension of state.

    2. Priori State Estimation.

        .. math::
            \mathbf{x}^{-}_{k} = f(\mathbf{x}_{k}, \mathbf{u}_{k}, t)

       where :math:`\mathbf{u}_{k}` is noise vector is randomly generated on the basis of the
       known PDF of :math:`\mathbf{u}`.

    3. Relative Likelihood.

        .. math::
            \begin{aligned}
                \mathbf{q}  = \mathbf{p} (\mathbf{y} |\mathbf{x}^{-}_{k}) \\
                \mathbf{q}_{i} = \frac{\mathbf{q}_{i}}{\sum_{j=1}^{N}\mathbf{q}_{j}}
            \end{aligned}

    4. Resample Particles.

       .. math::
           \begin{aligned}
               &\rule{113mm}{0.4pt}                                                      \\
               &\textbf{input}:\mathbf{x^{+}} (\text{State}), N (\text{number
               of particle}), \mathbf{q} (\text{relative likelihood})                    \\
               &\rule{113mm}{0.4pt}\\
               &\mathbf{sample} = [N] \\
               &\textbf{for} \: i=1 \: \textbf{to} \: \textbf{N}                         \\
               &\hspace{5mm} \mathbf{r}  = \mathbf{rand} (0,1)                           \\
               &\hspace{5mm} \textbf{for} \: j=1 \: \textbf{to} \: \textbf{N}            \\
               &\hspace{10mm} if \sum_{k=1}^{j}\mathbf{q} _{k}\ge r:                     \\
               &\hspace{15mm} \mathbf{sample}_{i} = \mathbf{x}^{+}_{j}                   \\
               &\rule{113mm}{0.4pt}                                               \\[-1.ex]
               &\bf{return} \:  \mathbf{sample}                                   \\[-1.ex]
               &\rule{113mm}{0.4pt}                                               \\[-1.ex]
           \end{aligned}

    5. Refine Posteriori And Covariances.

        .. math::

            \begin{aligned}
                \mathbf{x}^{+} =\frac{1}{N}  \sum_{i=1}^{n}\mathbf{sample}_{i}   \\
                P^{+} = \frac{1}{N} \sum_{i=1}^{N} (\mathbf{sample}_{i}-\mathbf{x}^{+})
                (\mathbf{sample}_{i}-\mathbf{x}^{+})^{T}
            \end{aligned}

    Example:
        1. Define a Nonlinear Time Invariant (NTI) system model

        >>> import torch, pypose as pp
        >>> class NTI(pp.module.System):
        ...     def __init__(self):
        ...         super().__init__()
        ...
        ...     def state_transition(self, state, input, t=None):
        ...         return state.cos() + input
        ...
        ...     def observation(self, state, input, t):
        ...         return state.sin() + input

        2. Create a model and filter

        >>> model = NTI()
        >>> pf = pp.module.PF(model)

        3. Prepare data

        >>> T, N = 5, 2 # steps, state dim
        >>> states = torch.zeros(T, N)
        >>> inputs = torch.randn(T, N)
        >>> observ = torch.zeros(T, N)
        >>> # std of transition, observation, and estimation
        >>> q, r, p = 0.1, 0.1, 10
        >>> Q = torch.eye(N) * q**2
        >>> R = torch.eye(N) * r**2
        >>> P = torch.eye(N).repeat(T, 1, 1) * p**2
        >>> estim = torch.randn(T, N) * p

        4. Perform PF prediction. Note that estimation error becomes smaller with more steps.

        >>> for i in range(T - 1):
        ...     w = q * torch.randn(N) # transition noise
        ...     v = r * torch.randn(N) # observation noise
        ...     states[i+1], observ[i] = model(states[i] + w, inputs[i])
        ...     estim[i+1], P[i+1] = pf(estim[i], observ[i] + v, inputs[i], P[i], Q, R)
        ... print('Est error:', (states - estim).norm(dim=-1))
        Est error: tensor([5.3627, 0.5640, 0.0953, 0.0447, 0.0936])

    Note:
        Implementation is based on Section 15.2 of this book

        * Dan Simon, `Optimal State Estimation: Kalman, H∞, and Nonlinear Approaches
          <https://onlinelibrary.wiley.com/doi/book/10.1002/0470045345>`_,
          Cleveland State University, 2006
    '''

    def __init__(self, model, Q=None, R=None, particles=1000):
        super().__init__(model, Q, R)
        self.particles = particles

    def forward(self, x, y, u, P, Q=None, R=None, t=None, n=None):
        r'''
        Performs one step estimation.

        Args:
            x (:obj:`Tensor`): estimated system state of previous step
            y (:obj:`Tensor`): system observation at current step (measurement)
            u (:obj:`Tensor`): system input at current step
            P (:obj:`Tensor`): state estimation covariance of previous step
            Q (:obj:`Tensor`, optional): covariance of system transition model
            R (:obj:`Tensor`, optional): covariance of system observation model
            n (:obj:`Tensor`, optional): covariance weight for randomly generate particles

        Return:
            list of :obj:`Tensor`: posteriori state and covariance estimation
        '''
        # Upper cases are matrices, lower cases are vectors

        Q = Q if Q is not None else self.Q
        R = R if R is not None else self.R
        n = x.size(-1) if n is None else n
        self.model.set_refpoint(state=x, input=u, t=t)

        xp = self.generate_particles(x, n * P)
        xs, ye = self.model(xp, u)
        q = self.relative_likelihood(y, ye, R)
        xr = self.resample_particles(q, xs)

        x = xr.mean(dim=-2)
        ex = xr - x
        P = self.compute_cov(ex, ex, Q)

        return x, P

    def generate_particles(self, x, P):
        r'''
        Randomly generate particles

        Args:
            x (:obj:`Tensor`): estimated system state of previous step
            P (:obj:`Tensor`): state estimation covariance of previous step

        Return:
            list of :obj:`Tensor`: particles
        '''
        m = MultivariateNormal(x, P)
        return m.sample(torch.Size([self.particles]))

    def relative_likelihood(self, y, ye, R):
        r'''
        Compute the relative likelihood
        '''
        return F.softmax(MultivariateNormal(ye, R).log_prob(y), dim=-1)

    def resample_particles(self, q, x):
        r'''
        Resample the set of a posteriori particles
        '''
        r = torch.rand(self.particles, dtype=x.dtype, device=x.device)
        cumsumq = torch.cumsum(q, dim=-1)
        return x[torch.searchsorted(cumsumq, r)]

    def compute_cov(self, a, b, Q=0):
        '''Compute covariance of two set of variables.'''
        return Q + bvv(a, b).mean(dim=-3)
