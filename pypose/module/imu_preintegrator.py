import torch
import pypose as pp
from torch import nn


class IMUPreintegrator(nn.Module):
    def __init__(self, position:torch.Tensor, rotation:pp.SO3, velocity:torch.Tensor):
        super().__init__()
        # Initial status of IMU: position, rotation, velocity
        self.register_buffer('position', position.clone())
        self.register_buffer('rotation', rotation.clone())
        self.register_buffer('velocity', velocity.clone())
        self.gravity = torch.tensor([0, 0, 9.81007])
        self.reset()

    def reset(self):
        self._dp = torch.zeros(3)
        self._dv = torch.zeros(3)
        self._dr = pp.identity_SO3()
        self._dt = torch.zeros(1)

    def forward(self, ang, acc, t, known_rotation:pp.SO3=None):
        """
        IMU Preintegration from angular rate (ang), linear acclearation (acc), duration (t)
        See Eq. A9, A10 in https://rpg.ifi.uzh.ch/docs/RSS15_Forster_Supplementary.pdf
        Known IMU rotation can be provided for better precisioin
        """
        dr = pp.so3(ang*t).Exp()
        if isinstance(known_rotation, pp.LieGroup):
            a = acc - known_rotation.Inv() @ self.gravity
        else:
            a = acc - (self.rotation * self._dr * dr).Inv() @ self.gravity
        self._dp = self._dp + self._dv * t + self._dr @ a * 0.5 * t**2
        self._dv = self._dv + self._dr @ a * t
        self._dr = self._dr * dr
        self._dt = self._dt + t
        return self.predict()

    def predict(self):
        """
        Propagated IMU status.
        See Eq. 38 in http://rpg.ifi.uzh.ch/docs/TRO16_forster.pdf
        """
        self.position.data = self.position + self.rotation @ self._dp + self.velocity * self._dt
        self.velocity.data = self.velocity + self.rotation @ self._dv
        self.rotation.data = self.rotation * self._dr
        self.reset()
        return self.position.clone(), self.rotation.clone(), self.velocity.clone()
