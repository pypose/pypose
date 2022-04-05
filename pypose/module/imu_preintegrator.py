import torch
import pypose as pp
from torch import nn


class IMUPreintegrator(nn.Module):
    r"""
    Applies preintegration over IMU input signals.

    IMU updates from duration (:math:`\delta t`), angular rate (:math:`\omega`),
    linear acceleration (:math:`\mathbf{a}`) in body frame, as well as their
    measurement covariance for angular rate :math:`C_{\omega}` and acceleration
    :math:`C_{\mathbf{a}}`. Known IMU rotation :math:`R` estimation can also be provided
    for better precision.

    Args:
        position (torch.Tensor, optional): initial postion. Default: torch.zeros(3)
        rotation (pypose.SO3, optional): initial rotation. Default: :meth:`pypose.identity_SO3`
        velocity (torch.Tensor, optional): initial postion. Default: torch.zeros(3)
        gravity (float, optional): the gravity acceleration. Default: 9.81007
    """
    def __init__(self, position = torch.zeros(3),
                       rotation = pp.identity_SO3(),
                       velocity = torch.zeros(3),
                       gravity = 9.81007):
        super().__init__()
        # Initial status of IMU: (pos)ition, (rot)ation, (vel)ocity, (cov)ariance
        self.register_buffer('gravity', torch.tensor([0, 0, gravity]))
        self.register_buffer('pos', position.clone())
        self.register_buffer('rot', rotation.clone())
        self.register_buffer('vel', velocity.clone())
        self.register_buffer('cov', torch.zeros(9,9))
        # Note that cov[9x9] order is rot, vel, pos
        self.register_buffer('_dp', torch.zeros(3))
        self.register_buffer('_dv', torch.zeros(3))
        self.register_buffer('_dr', pp.identity_SO3())
        self.register_buffer('_dt', torch.zeros(1))

    def reset(self):
        self._dp.zero_()
        self._dv.zero_()
        self._dr.identity_()
        self._dt.zero_()

    def update(self, dt, ang, acc, rot:pp.SO3=None, ang_cov=None, acc_cov=None):
        r"""
        IMU Preintegration from duration (dt), angular rate (ang), linear acceleration (acc)
        Uncertainty propagation from measurement covariance (cov): ang_cov, acc_cov
        Known IMU rotation (rot) estimation can be provided for better precision

        Uncertainty Propagation:

        .. math::
            \begin{align*}
                B &= [B_g, B_a]                              \\
                C &= A C A^T + B \mathrm{diag}(C_g, C_a) B^T \\
                  &= A C A^T + B_g C_g B_g^T + B_a C_a B_a^T
            \end{align*}

        Args:
            dt (torch.Tensor): time interval from last update. :code:`shape`: (1)
            ang (torch.Tensor): angular rate (:math:`\omega`) in IMU body frame. :code:`shape`: (3)
            acc (torch.Tensor): linear acceleration (:math:`\mathbf{a}`) in IMU body frame.
                :code:`shape`: (3)
            rot (pypose.SO3, optional): known IMU rotation. :code:`lshape`: (1)
            ang_cov (torch.Tensor, optional): covariance matrix of angular rate. :code:`shape`: (3, 3).
                Default: :code:`torch.eye(3)*(1.6968*10**-4)**2` (Adapted from Euroc dataset)
            acc_cov (torch.Tensor, optional): covariance matrix of linear acceleration.
                :code:`shape`: (3, 3). Default: :code:`torch.eye(3)*(2.0*10**-3)**2`  (Adapted from
                Euroc dataset)

        Refer to Eq. (A9), (A10), (A7), (A8) in
        `this RSS paper <https://rpg.ifi.uzh.ch/docs/RSS15_Forster_Supplementary.pdf>`_ for more details.
        """
        dr = pp.so3(ang*dt).Exp()
        if isinstance(rot, pp.LieTensor):
            a = acc - rot.Inv() @ self.gravity
        else:
            a = acc - (self.rot * self._dr * dr).Inv() @ self.gravity
        self._dp = self._dp + self._dv * dt + self._dr @ a * 0.5 * dt**2
        self._dv = self._dv + self._dr @ a * dt
        self._dr = self._dr * dr
        self._dt = self._dt + dt

        if ang_cov is None: # gyro covariance
            Cg = torch.eye(3, device=dt.device) * (1.6968*10**-4)**2
        if acc_cov is None: # acc covariance
            Ca = torch.eye(3, device=dt.device) * (2.0*10**-3)**2

        Ha = pp.vec2skew(acc)
        A = torch.eye(9, device=dt.device)
        A[0:3, 0:3] = dr.matrix().mT
        A[3:6, 0:3] = - self._dr.matrix() @ Ha * dt
        A[6:9, 0:3] = - 0.5 * self._dr.matrix() @ Ha * dt**2
        A[6:9, 3:6] = torch.eye(3, device=dt.device) * dt

        Bg = torch.zeros(9, 3, device=dt.device)
        Bg[0:3, 0:3] = pp.so3(ang*dt).Jr() * dt
        Ba = torch.zeros(9, 3, device=dt.device)
        Ba[3:6, 0:3] = self._dr.matrix() * dt
        Ba[6:9, 0:3] = 0.5 * self._dr.matrix() * dt**2

        self.cov = A @ self.cov @ A.mT + Bg @ Cg @ Bg.mT / dt + Ba @ Ca @ Ba.mT / dt

    def forward(self, reset=True):
        r"""
        Propagated IMU status.

        Args:
            reset (bool, optional): if reset the preintegrator to initial state. Default: :code:`False`

        Returns:
            :code:`dict`: A :class:`dict` containing 4 items: 'rot'ation, 'vel'ocity, 'pos'ition, and 'cov'ariance.

            - 'rot' (pypose.SO3): rotation. :code:`lshape`: (1)

            - 'vel' (torch.Tensor): velocity. :code:`shape`: (3)

            - 'pos' (torch.Tensor): postion. :code:`shape`: (3)

            - 'cov' (torch.Tensor): covariance (order: rotation, velocity, position). :code:`shape`: (9, 9)

        Note:
            Output covariance (Shape: (9, 9)) is in the order of rotation, velocity, and position.

        Refer to Eq. (38) in `this TRO paper <http://rpg.ifi.uzh.ch/docs/TRO16_forster.pdf>`_ for more details.
        """
        self.pos = self.pos + self.rot @ self._dp + self.vel * self._dt
        self.vel = self.vel + self.rot @ self._dv
        self.rot = self.rot * self._dr
        if reset is True:
            self.reset()
        return {'rot':self.rot.clone(),
                'vel':self.vel.clone(),
                'pos':self.pos.clone(),
                'cov':self.cov.clone()}
