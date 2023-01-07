import torch
from torch import nn
from ..basics import bmv
from torch.linalg import pinv
from scipy import stats
import numpy as np


class PF(nn.Module):
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

    Warning:
        Don't introduce noise in ``System`` methods ``state_transition`` and ``observation``
        for filter testing, as those methods are used for automatically linearizing the system
        by the parent class ``pypose.module.System``, unless your system model explicitly
        introduces noise.

    Note:
        Implementation is based on Section 15.3 of this book

        * Dan Simon, `Optimal State Estimation: Kalman, H∞, and Nonlinear Approaches
          <https://onlinelibrary.wiley.com/doi/epdf/10.1002/0470045345.fmatter>`_,
          Cleveland State University, 2006
    '''

    def __init__(self, model, Q=None, R=None, particle_number=100):
        super().__init__()
        self.set_uncertainty(Q=Q, R=R)
        self.model = model
        self.particle_number = particle_number

    def generate_particle(self, x, Q):
        r'''
        randomly generate particle
        '''

        x_sample = torch.distributions.MultivariateNormal(x, Q).sample(
            (self.particle_number,))

        return x_sample

    def relative_likelihood(self, y_estimate, y, R):
        r'''
        Compute the relative likelihood
        '''

        rv = stats.multivariate_normal(y, R)
        q = rv.pdf(y_estimate) + 0.01
        q = q / np.sum(q)

        return torch.from_numpy(q)

    def refine_particles(self, q, x):
        r'''
        Refine the set of a posteriori particles
        '''

        x_new = torch.zeros(self.particle_number, x.shape[1])
        for n in range(self.particle_number):
            r = torch.rand(1)[0]
            sum_q = 0.

            for j in range(self.particle_number):
                sum_q += q[j]
                if sum_q >= r:
                    x_new[n] = x[j]
                    break

        return x_new

    def compute_conv(self, x_estimate, x_particle):
        r'''
        compute covariance

        Args:
            x_estimate (:obj:`Tensor`): System estimate.
            x_particle (:obj:`Tensor`):  Particle of estimate.

        Return:
            out (:obj:`Tensor`): the covariance of  prior state estimation .
        '''

        e = torch.sub(x_particle, x_estimate).unsqueeze(2)
        p_estimate = torch.sum(torch.bmm(e, e.permute(0, 2, 1)),
                               dim=0) / (self.particle_number - 1)

        return p_estimate

    def forward(self, x, y, u, P, Q=None, R=None):
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
        self.model.set_refpoint(state=x, input=u)
        I = torch.eye(P.shape[-1], device=P.device, dtype=P.dtype)
        A, B = self.model.A, self.model.B
        C, D = self.model.C, self.model.D
        Q = Q if Q is not None else self.Q
        R = R if R is not None else self.R

        # PF
        x_particle = self.generate_particle(x, Q)
        x = self.model.state_transition(x_particle, u)  # 1. System transition
        y_estimate = self.model.observation(x, u)

        P = A @ P @ A.mT + Q  # 2. Covariance predict
        K = P @ C.mT @ pinv(C @ P @ C.mT + R)  # 3. Kalman gain
        e = y - y_estimate  # predicted observation error
        x = x + bmv(K, e)  # 4. Posteriori state

        # resample
        q = self.relative_likelihood(y_estimate, y, R)
        x_resample = self.refine_particles(q, x)

        # update state and  Posteriori covariance
        x = x_resample.mean(dim=0)
        P = self.compute_conv(x, x_resample)
        P = (I - K @ C) @ P  # 5. Posteriori covariance

        return x, P

    @property
    def Q(self):
        r'''
        The covariance of system transition noise.
        '''
        if not hasattr(self, '_Q'):
            raise NotImplementedError('Call set_uncertainty() to define\
                                        transition covariance Q.')
        return self._Q

    @property
    def R(self):
        r'''
        The covariance of system observation noise.
        '''
        if not hasattr(self, '_R'):
            raise NotImplementedError('Call set_uncertainty() to define\
                                        transition covariance R.')
        return self._R

    def set_uncertainty(self, Q=None, R=None):
        r'''
        Set the covariance matrices of transition noise and observation noise.

        Args:
            Q (:obj:`Tensor`): batched square covariance matrices of transition noise.
            R (:obj:`Tensor`): batched square covariance matrices of observation noise.
        '''
        if Q is not None:
            self.register_buffer("_Q", Q)
        if R is not None:
            self.register_buffer("_R", R)
