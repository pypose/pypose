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
                       ang_cov = (1.6968*10**-4)**2,
                       acc_cov = (2.0*10**-3)**2,
                       prop_cov = True,
                       reset = False):
        super().__init__()
        self.reset, self.prop_cov = reset, prop_cov
        # Initial status of IMU: (pos)ition, (rot)ation, (vel)ocity, (cov)ariance
        self.gravity, self.ang_cov, self.acc_cov = gravity, ang_cov, acc_cov
        self.register_buffer('cov', torch.zeros(1, 9, 9))
        self.register_buffer('pos', pos.clone())
        self.register_buffer('rot', rot.clone())
        self.register_buffer('vel', vel.clone())

    def forward(self, dt, ang, acc, rot:pp.SO3=None, cov=None, init_state=None, ang_cov=None, acc_cov=None):
        '''
        Args:
            acc: (B, Frames, 3)
            ang: (B, Frames, 3)
            dt: (B, Frames, 1)
            rot: (B, Frames, 4)
            init_state: dict

        dr, dp, dv: represents the incremental of one small interval 
        incre_r, incre_p, incre_v: represents the incremental starting from the first frame
        '''
        assert(len(acc.shape) == len(dt.shape) == len(ang.shape) == 3)

        if self.reset is False and init_state is None:
            init_state = {'pos': self.pos, 'rot': self.rot, 'vel': self.vel}

        integrate = self.integrate(init_state, dt, ang, acc, rot)
        predict   = self.predict(init_state, integrate)

        if self.prop_cov:
            if ang_cov is None:
                ang_cov = self.ang_cov
            if acc_cov is None:
                acc_cov = self.acc_cov
            if cov is None:
                cov = self.cov
            cov = self.propagate_cov(dt=dt, acc=acc, integrate=integrate, init_cov=cov, ang_cov=ang_cov, acc_cov=acc_cov)
        else:
            cov = {'cov': None}

        self.pos = predict['pos'][..., -1:, :]
        self.rot = predict['rot'][..., -1:, :]
        self.vel = predict['vel'][..., -1:, :]
        self.cov = cov['cov']

        return {**predict, **cov}

    def integrate(self, init_state, dt, ang, acc, rot:pp.SO3=None):
        B, F = dt.shape[:2]
        gravity = torch.tensor([0, 0, self.gravity], dtype = dt.dtype, device = dt.device)
        dr = torch.cat([pp.identity_SO3(B,1,dtype=dt.dtype, device=dt.device), pp.so3(ang*dt).Exp()], dim=1)
        incre_r = pp.cumprod(dr, dim = 1, left=False)
        inte_rot = init_state['rot'] * incre_r

        if isinstance(rot, pp.LieTensor):
            assert(len(rot.shape) == 3)
            a = acc - rot.Inv() @ gravity
        else:
            a = acc - inte_rot[:,1:,:].Inv() @ gravity

        dv = torch.zeros(B,1,3,dtype=dt.dtype, device=dt.device)
        dv = torch.cat([dv, incre_r[:,:F,:] @ a * dt], dim=1)
        incre_v = torch.cumsum(dv, dim =1)

        dp = torch.zeros(B,1,3,dtype=dt.dtype, device=dt.device)
        dp = torch.cat([dp, incre_v[:,:F,:] * dt + incre_r[:,:F,:] @ a * 0.5 * dt**2], dim =1)
        incre_p = torch.cumsum(dp, dim =1)

        incre_t = torch.cumsum(dt, dim = 1)
        incre_t = torch.cat([torch.zeros(B,1,1,dtype=dt.dtype, device=dt.device), incre_t], dim =1)

        return {'vel':incre_v[...,1:,:], 'pos':incre_p[:,1:,:], 'rot':incre_r[:,1:,:],
                'dt':incre_t[...,1:,:], 'dr': dr[:,1:,:]}

    @classmethod
    def predict(cls, init_state, integrate):
        return {
            'rot': init_state['rot'] * integrate['rot'],
            'vel': init_state['vel'] + init_state['rot'] * integrate['vel'],
            'pos': init_state['pos'] + init_state['rot'] * integrate['pos'] + init_state['vel'] * integrate['dt'],
        }

    @classmethod
    def propagate_cov(cls, dt, acc, integrate, init_cov, ang_cov, acc_cov):
        '''
            Covariance propogation
        '''
        B, F = dt.shape[:2]
        Cg = torch.eye(3, device=dt.device, dtype=dt.dtype) * ang_cov
        Cg = Cg.repeat([B, F, 1, 1])
        Ca = torch.eye(3, device=dt.device, dtype=dt.dtype) * acc_cov
        Ca = Ca.repeat([B, F, 1, 1])

        if init_cov is None:
            init_cov = torch.zeros((B, 9, 9), device=dt.device, dtype=dt.dtype)

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
        B_cov = torch.einsum('...xy,...t -> ...xy', Bg @ Cg @ Bg.mT + Ba @ Ca @ Ba.mT, 1/dt)
        B_cov = torch.cat([init_cov[:,None,...], B_cov], dim=1)

        A_left_cum = pp.cumprod(A.flip([1]), dim=1).flip([1]) # cumfrom An to I, then flip
        A_right_cum = A_left_cum.mT
        
        cov = (A_left_cum @ B_cov @ A_right_cum).sum(dim=1)

        return {'cov': cov}
