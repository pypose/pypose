import torch
import pypose as pp
from torch import nn


class IMUPreintegrator(nn.Module):
    r'''
    Applies preintegration over IMU input signals.

    IMU updates from duration (:math:`\delta t`), angular rate (:math:`\omega`),
    linear acceleration (:math:`\mathbf{a}`) in body frame, as well as their
    measurement covariance for angular rate :math:`C_{g}` and acceleration
    :math:`C_{\mathbf{a}}`. Known IMU rotation :math:`R` estimation can also be provided
    for better precision.

    Args:
        pos (torch.Tensor, optional): initial postion. Default: torch.zeros(3)
        rot (pypose.SO3, optional): initial rotation. Default: :meth:`pypose.identity_SO3`
        vel (torch.Tensor, optional): initial postion. Default: torch.zeros(3)
        gravity (float, optional): the gravity acceleration. Default: 9.81007
    '''
    def __init__(self, pos = torch.zeros(3),
                       rot = pp.identity_SO3(),
                       vel = torch.zeros(3),
                       gravity = 9.81007,
                       gyro_cov = (1.6968e-4)**2,
                       acc_cov = (2e-3)**2,
                       prop_cov = True,
                       reset = True):
        super().__init__()
        self.reset, self.prop_cov = reset, prop_cov
        # Initial status of IMU: (pos)ition, (rot)ation, (vel)ocity, (cov)ariance
        self.gravity, self.gyro_cov, self.acc_cov = gravity, gyro_cov, acc_cov
        self.cov = torch.zeros(1, 9, 9)
        self.pos = pos.clone()
        self.rot = rot.clone()
        self.vel = vel.clone()

    def forward(self, dt, gyro, acc, rot:pp.SO3=None, gyro_cov=None, acc_cov=None, init_state=None):
        '''
        Args:
            acc: (B, Frames, 3)
            gyro: (B, Frames, 3)
            dt: (B, Frames, 1)
            rot: (B, Frames, 4)
            init_state: dict

        dr, dp, dv: represents the incremental of one small interval 
        incre_r, incre_p, incre_v: represents the incremental starting from the first frame
        '''
        assert(len(acc.shape) == len(dt.shape) == len(gyro.shape) == 3)

        if self.reset is False and init_state is None:
            init_state = {'pos': self.pos, 'rot': self.rot, 'vel': self.vel}

        integrate = self.integrate(init_state, dt, gyro, acc, rot)
        predict   = self.predict(init_state, integrate)

        if self.prop_cov:
            if gyro_cov is None:
                gyro_cov = self.gyro_cov
            if acc_cov is None:
                acc_cov = self.acc_cov
            if 'cov' not in init_state or init_state['cov'] is None:
                init_cov = self.cov
            else:
                init_cov = init_state['cov']
            cov = self.propagate_cov(dt, acc, integrate, init_cov, gyro_cov, acc_cov)
        else:
            cov = {'cov': None}

        self.pos = predict['pos'][..., -1:, :]
        self.rot = predict['rot'][..., -1:, :]
        self.vel = predict['vel'][..., -1:, :]
        self.cov = cov['cov']

        return {**predict, **cov}

    def integrate(self, init_state, dt, gyro, acc, rot:pp.SO3=None):
        B, F = dt.shape[:2]
        gravity = torch.tensor([0, 0, self.gravity], dtype = dt.dtype, device = dt.device)
        dr = torch.cat([pp.identity_SO3(B, 1, dtype=dt.dtype, device=dt.device), pp.so3(gyro*dt).Exp()], dim=1)
        incre_r = pp.cumprod(dr, dim = 1, left=False)
        inte_rot = init_state['rot'] * incre_r

        if isinstance(rot, pp.LieTensor):
            assert(len(rot.shape) == 3)
            a = acc - rot.Inv() @ gravity
        else:
            a = acc - inte_rot[:,1:,:].Inv() @ gravity

        dv = torch.zeros(B, 1, 3, dtype=dt.dtype, device=dt.device)
        dv = torch.cat([dv, incre_r[:,:F,:] @ a * dt], dim=1)
        incre_v = torch.cumsum(dv, dim =1)

        dp = torch.zeros(B, 1, 3, dtype=dt.dtype, device=dt.device)
        dp = torch.cat([dp, incre_v[:,:F,:] * dt + incre_r[:,:F,:] @ a * 0.5 * dt**2], dim =1)
        incre_p = torch.cumsum(dp, dim =1)

        incre_t = torch.cumsum(dt, dim = 1)
        incre_t = torch.cat([torch.zeros(B, 1, 1, dtype=dt.dtype, device=dt.device), incre_t], dim =1)

        return {'vel':incre_v[...,1:,:], 'pos':incre_p[:,1:,:], 'rot':incre_r[:,1:,:],
                'dt':incre_t[...,1:,:], 'dr': dr[:,1:,:]}

    @classmethod
    def predict(cls, init_state, integrate):
        return {
            'rot': init_state['rot'] * integrate['rot'],
            'vel': init_state['vel'] + init_state['rot'] * integrate['vel'],
            'pos': init_state['pos'] + init_state['rot'] * integrate['pos'] + init_state['vel'] * integrate['dt'],
        }

    def propagate_cov(self, dt, acc, integrate, init_cov, gyro_cov, acc_cov):
        '''
            Covariance propogation
        '''
        B, F = dt.shape[:2]
        Cg = torch.eye(3, device=dt.device, dtype=dt.dtype) * gyro_cov
        Cg = Cg.repeat([B, F, 1, 1])
        Ca = torch.eye(3, device=dt.device, dtype=dt.dtype) * acc_cov
        Ca = Ca.repeat([B, F, 1, 1])

        Ha = pp.vec2skew(acc)
        A = torch.eye(9, device=dt.device, dtype=dt.dtype).repeat([B, F+1, 1, 1])
        A[:, :-1, 0:3, 0:3] = integrate['dr'].matrix().mT
        A[:, :-1, 3:6, 0:3] = torch.einsum('...xy,...t -> ...xy', \
            - integrate['rot'].matrix() @ Ha, dt)
        A[:, :-1, 6:9, 0:3] = torch.einsum('...xy,...t -> ...xy', \
            - 0.5 * integrate['rot'].matrix() @ Ha, dt**2)
        A[:, :-1, 6:9, 3:6] = torch.einsum('...xy,...t -> ...xy', \
            torch.eye(3, device=dt.device, dtype=dt.dtype).repeat([B, F, 1, 1]), dt)

        Bg = torch.zeros(B, F, 9, 3, device=dt.device, dtype=dt.dtype)
        Bg[..., 0:3, 0:3] = torch.einsum('...xy,...t -> ...xy', integrate['dr'].Log().Jr(), dt)

        Ba = torch.zeros(B, F, 9, 3, device=dt.device, dtype=dt.dtype)
        Ba[..., 3:6, 0:3] = torch.einsum('...xy,...t -> ...xy', integrate['rot'].matrix(), dt)
        Ba[..., 6:9, 0:3] = 0.5 * torch.einsum('...xy,...t -> ...xy', integrate['dr'].matrix(), dt**2)

        # the term of B
        init_cov = init_cov.expand(B, 9, 9)
        B_cov = torch.einsum('...xy,...t -> ...xy', Bg @ Cg @ Bg.mT + Ba @ Ca @ Ba.mT, 1/dt)
        B_cov = torch.cat([init_cov[:,None,...], B_cov], dim=1)

        A_left_cum = pp.cumprod(A.flip([1]), dim=1).flip([1]) # cumfrom An to I, then flip
        A_right_cum = A_left_cum.mT

        cov = (A_left_cum @ B_cov @ A_right_cum).sum(dim=1)

        return {'cov': cov}
