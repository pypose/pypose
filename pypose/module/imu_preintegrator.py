import torch
import pypose as pp
from torch import nn


class IMUPreintegrator(nn.Module):
    def __init__(self, position:torch.Tensor, rotation:pp.SO3, velocity:torch.Tensor, gravity=9.81007):
        super().__init__()
        # Initial status of IMU: position, rotation, velocity
        self.register_buffer('gravity', torch.tensor([0, 0, gravity]))
        self.register_buffer('position', position.clone())
        self.register_buffer('rotation', rotation.clone())
        self.register_buffer('velocity', velocity.clone())
        self.register_buffer('_dp', torch.zeros(3))
        self.register_buffer('_dv', torch.zeros(3))
        self.register_buffer('_dr', pp.identity_SO3())
        self.register_buffer('_dt', torch.zeros(1))

    def reset(self):
        self._dp.zero_()
        self._dv.zero_()
        self._dr.identity_()
        self._dt.zero_()

    def update(self, dt, ang, acc, rotation:pp.SO3=None):
        """
        IMU Preintegration from duration (dt), angular rate (ang), linear acclearation (acc)
        Known IMU rotation (estimation) can be provided for better precision
        See Eq. A9, A10 in https://rpg.ifi.uzh.ch/docs/RSS15_Forster_Supplementary.pdf
        """
        dr = pp.so3(ang*dt).Exp()
        if isinstance(rotation, pp.LieGroup):
            a = acc - rotation.Inv() @ self.gravity
        else:
            a = acc - (self.rotation * self._dr * dr).Inv() @ self.gravity
        self._dp = self._dp + self._dv * dt + self._dr @ a * 0.5 * dt**2
        self._dv = self._dv + self._dr @ a * dt
        self._dr = self._dr * dr
        self._dt = self._dt + dt

    def forward(self, reset=True):
        """
        Propagated IMU status.
        See Eq. 38 in http://rpg.ifi.uzh.ch/docs/TRO16_forster.pdf
        """
        self.position = self.position + self.rotation @ self._dp + self.velocity * self._dt
        self.velocity = self.velocity + self.rotation @ self._dv
        self.rotation = self.rotation * self._dr
        if reset is True:
            self.reset()
        return self.position.clone(), self.rotation.clone(), self.velocity.clone()
