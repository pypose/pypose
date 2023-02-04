import torch
from ..basics import bmv
from torch.linalg import pinv
from pypose.module import EKF


class PF(EKF):
    r'''
    Performs Batched Particle_Filters (PF).

    Args:
        model (:obj:`System`): The system model to be estimated, a subclass of
            :obj:`pypose.module.System`.
        Q (:obj:`Tensor`, optional): The covariance matrices of system transition noise.
            Ignored if provided during each iteration. Default: ``None``
        R (:obj:`Tensor`, optional): The covariance matrices of system observation noise.
            Ignored if provided during each iteration. Default: ``None``
        particle_number (:obj:`Int`, optional): The number of particle. Default: ``1000``

    A non-linear system can be described as

    .. math::
        \begin{aligned}
            \mathbf{x}_{k+1} &= \mathbf{f}(\mathbf{x}_k, \mathbf{u}_k, t_k) + \mathbf{w}_k,
            \quad \mathbf{w}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{Q})  \\
            \mathbf{y}_{k} &= \mathbf{g}(\mathbf{x}_k, \mathbf{u}_k, t_k) + \mathbf{v}_k,
            \quad \mathbf{v}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{R})
        \end{aligned}

    PF can be described as the following equations, where the subscript :math:`\cdot_{k}`
    is omited for simplicity.

    1. Generate Particles.

        .. math::
            \begin{aligned}
                \mathbf{x} _{k} = \mathbf{p} (\mathbf{x} ,N) \quad k=1,...,N \\
                \mathbf{P} _{k} = \mathbf{P} \quad k=1,...,N
            \end{aligned}

    where :math:`N` is the number of Particles and :math:`\mathbf{p}` is the probability
    density function.

    2. Priori State Estimation.

        .. math::
            \mathbf{x}^{-}_{k} = f(\mathbf{x}_{k}, \mathbf{u}_{k}, t)

    where :math:`\mathbf{u}_{k}` is noise vector is randomly generated on the basis of the known
    pdf of :math:`\mathbf{u}`.

    3. Priori Covariance Propagation.

        .. math::
            \mathbf{P}^{-}_{k} = \mathbf{A}\mathbf{P}_{k}\mathbf{A}^{T} + \mathbf{Q}

    4. Update Kalman Gain

        .. math::
            \begin{aligned}
                \mathbf{C} _{k} = \frac{∂\mathbf{g} }{∂\mathbf{x} }|_{x=\mathbf{x}^{-}_{k}} \\
                \mathbf{K}_{k} = \mathbf{P}\mathbf{C}^{T}_{k}
                (\mathbf{C}_{k}\mathbf{P} \mathbf{C}^{T}_{k} + \mathbf{R})^{-1}
            \end{aligned}

    5. Posteriori State Estimation.

        .. math::
            \mathbf{x_{k}}^{+} = \mathbf{x}^{-} + \mathbf{K}\left(\mathbf{y} -
            \mathbf{g}(\mathbf{x}_{k},\mathbf{u}_{k},t)\right)

    6. Posteriori Covariance Estimation.

        .. math::
            \mathbf{P_{k}} = (\mathbf{I} - \mathbf{K}_{k}\mathbf{C}_{k}) \mathbf{P}^{-}_{k}

    7. Relative Likelihood

        .. math::
            \begin{aligned}
                \mathbf{q}  = \mathbf{p} (\mathbf{y} |\mathbf{x}^{+}_{k}) \\
                \mathbf{q}_{i} = \frac{\mathbf{q}_{i}}{\sum_{j=1}^{N}\mathbf{q}_{j}}
            \end{aligned}

    8. Resample Particles

        .. math::
            \begin{aligned}
                &\rule{113mm}{0.4pt}                                                  \\ \\
                &\textbf{input}:\mathbf{x^{+}}({\tiny State} ) ,N( {\tiny number
                \quad of\quad particle} ),\mathbf{q}({\tiny relative\quad likelihood} )   \\
                &\rule{113mm}{0.4pt}\\
                &\mathbf{sample} = [N] \\
                &\textbf{for} \: i=1 \: \textbf{to} \: \textbf{N}                         \\
                &\hspace{5mm} \mathbf{r}  = \mathbf{rand} (0,1)                           \\
                &\hspace{5mm} \textbf{for} \: j=1 \: \textbf{to} \: \textbf{N}            \\
                &\hspace{10mm} if \sum_{k=1}^{j}\mathbf{q} _{k}\ge r:                     \\
                &\hspace{15mm} \mathbf{sample}_{i} = \mathbf{x}^{+}_{j}
                &\rule{113mm}{0.4pt}                                                 \\[-1.ex]
                &\bf{return} \:  \mathbf{sample}                                     \\[-1.ex]
                &\rule{113mm}{0.4pt}                                                 \\[-1.ex]
            \end{aligned}

    9. Refine Posteriori And Covariances

        .. math::
            \begin{aligned}
               \mathbf{x}^{+} =\frac{1}{N}  \sum_{i=1}^{n}\mathbf{sample}_{i}   \\
               P^{+} = \frac{1}{N-1} \sum_{i=1}^{N} (\mathbf{sample}_{i}-\mathbf{x}^{+})(
               \mathbf{sample}_{i}-\mathbf{x}^{+})^{T}
            \end{aligned}

    Note:
        Implementation is based on Section 15.3 of this book

        * Dan Simon, `Optimal State Estimation: Kalman, H∞, and Nonlinear Approaches
          <https://onlinelibrary.wiley.com/doi/epdf/10.1002/0470045345.fmatter>`_,
          Cleveland State University, 2006
    '''

    def __init__(self, model, Q=None, R=None, particle_number=None):
        super().__init__(model, Q, R)
        self.particle_number = 1000 if particle_number is None else particle_number

    def forward(self, x, y, u, P, Q=None, R=None, t=None):
        r'''
        Performs one step estimation.

        Args:
            x (:obj:`Tensor`): estimated system state of previous step
            y (:obj:`Tensor`): system observation at current step (measurement)
            u (:obj:`Tensor`): system input at current step
            P (:obj:`Tensor`): state estimation covariance of previous step
            Q (:obj:`Tensor`, optional): covariance of system transition model
            R (:obj:`Tensor`, optional): covariance of system observation model

        Return:
            list of :obj:`Tensor`: posteriori state and covariance estimation
        '''
        # Upper cases are matrices, lower cases are vectors
        self.model.set_refpoint(state=x, input=u, t=t)
        A, B = self.model.A, self.model.B
        C, D = self.model.C, self.model.D
        Q = Q if Q is not None else self.Q
        R = R if R is not None else self.R

        xp = self.generate_particle(x, P)
        x = self.model.state_transition(xp, u, t)  # 1. System transition
        ye = self.model.observation(x, u)
        P = A @ P @ A.mT + Q  # 2. Covariance predict
        # self.model.set_refpoint(state=x, input=u, t=t)
        # C1 = self.model.C   #not work
        K = P @ C.mT @ pinv(C @ P @ C.mT + R)  # 3. Kalman gain
        e = y - ye  # predicted observation error
        x = x + bmv(K, e)  # 4. Posteriori state

        # resample
        q = self.relative_likelihood(ye, y, R)
        xr = self.resample_particles(q, x)

        # update state and  Posteriori covariance
        x = xr.mean(dim=0)
        ex = xr - x
        P = self.compute_cov(ex, ex) / (self.particle_number - 1)
        return x, P

    def generate_particle(self, x, P):
        r'''
        Randomly generate particles
        '''
        xp = torch.distributions.MultivariateNormal(x, P).sample((self.particle_number,))
        return xp

    def relative_likelihood(self, ye, y, R):
        r'''
        Compute the relative likelihood
        '''
        q = torch.distributions.MultivariateNormal(y, R).log_prob(ye).exp() + 1.e-2
        q = q / torch.sum(q)
        return q

    def resample_particles(self, q, x):
        r'''
        Resample the set of a posteriori particles
        '''
        r = torch.rand(self.particle_number, device=x.device)
        cumsumq = torch.cumsum(q, dim=0)
        return x[torch.searchsorted(cumsumq, r)]

    def compute_cov(self, a, b, Q=0):
        '''Compute covariance of two set of variables.'''
        a, b = a.unsqueeze(-1), b.unsqueeze(-1)
        return Q + (a @ b.mT).sum(dim=-3)
