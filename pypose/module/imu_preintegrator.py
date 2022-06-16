import torch
import pypose as pp
from torch import nn


class IMUPreintegrator(nn.Module):
    r"""
    Applies preintegration over IMU input signals.

    IMU updates from duration (:math:`\delta t`), angular rate (:math:`\omega`),
    linear acceleration (:math:`\mathbf{a}`) in body frame, as well as their
    measurement covariance for angular rate :math:`C_{g}` and acceleration
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
                       gravity = 9.81007,
                       ang_cov =  (1.6968*10**-4)**2,
                       acc_cov = (2.0*10**-3)**2,
                       update = False):
        super().__init__()
        self.update = update
        # Initial status of IMU: (pos)ition, (rot)ation, (vel)ocity, (cov)ariance
        self.gravity, self.ang_cov, self.acc_cov = gravity, ang_cov, acc_cov
        
        self.register_buffer('pos', position.clone())
        self.register_buffer('rot', rotation.clone())
        self.register_buffer('vel', velocity.clone())
        self.register_buffer('cov', torch.zeros(1,9,9))

    def forward(self, dt, ang, acc, rot:pp.SO3=None, init_state = None, cov_propogation = True):
        assert(len(acc.shape) == 3)
        assert(len(dt.shape) == 3)
        assert(len(ang.shape) == 3)

        if self.update is True and init_state is None:
            init_state = {
                'p': self.pos,
                'r': self.rot,
                'v': self.vel,
            }
        
        inte_state, cov_state = self.batch_imu_integrate(init_state=init_state, dt=dt, ang=ang, acc=acc, rot=rot, cov=self.cov, 
                                                            gravity=self.gravity, cov_propogation=cov_propogation)

        if self.update:
            self.pos, self.rot, self.vel = inte_state['pos'][..., -1:, :], inte_state['rot'][..., -1:, :], inte_state['vel'][..., -1:, :]
            self.cov = cov_state['cov']

        return inte_state, cov_state

    @classmethod
    def batch_imu_update(self, init_state, dt, ang, acc, rot:pp.SO3=None, gravity=9.81007):

        B, F = dt.shape[:2]
        gravity = torch.tensor([0, 0, gravity], dtype = dt.dtype, device = dt.device)
        dr = torch.cat([pp.identity_SO3(B,1,dtype=dt.dtype, device=dt.device), pp.so3(ang*dt).Exp()], dim=1)
        incre_r = pp.cumprod(dr, dim = 1, left=False)
        inte_rot = init_state["r"] * incre_r

        if isinstance(rot, pp.LieTensor):
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

        return {"incre_v":incre_v[...,1:,:], "incre_p":incre_p[:,1:,:], "incre_r":incre_r[:,1:,:], 
                "incre_t":incre_t[...,1:,:], "dr": dr[:,1:,:]}

    @classmethod
    def cov_propogate(self, dt, acc, dr, incre_r, init_cov=None, ang_cov=(1.6968*10**-4)**2, acc_cov=(2.0*10**-3)**2):
        """
        Batched covariance propogation
        """
        B = dt.shape[0]

        Cg = torch.eye(3, device=dt.device, dtype=dt.dtype) * ang_cov
        Cg = Cg.repeat([B, 1, 1])
        Ca = torch.eye(3, device=dt.device, dtype=dt.dtype) * acc_cov
        Ca = Ca.repeat([B, 1, 1])

        if init_cov is None:
            init_cov = torch.zeros((B,9,9), device=dt.device, dtype=dt.dtype)

        Ha = pp.vec2skew(acc) # (B, 3, 3)

        A = torch.eye(9, device=dt.device, dtype=dt.dtype).repeat([B, 1, 1])
        A[..., 0:3, 0:3] = dr.matrix().mT
        A[:, 3:6, 0:3] = torch.einsum('bxy,bt -> bxy', - incre_r.matrix() @ Ha, dt)
        A[:, 6:9, 0:3] = torch.einsum('bxy,bt -> bxy', - 0.5 * incre_r.matrix() @ Ha, dt**2)
        A[:, 6:9, 3:6] = torch.einsum('bxy,bt -> bxy', torch.eye(3, device=dt.device, dtype=dt.dtype).repeat([B, 1, 1]), dt)

        Bg = torch.zeros(B, 9, 3, device=dt.device, dtype=dt.dtype)
        Bg[..., 0:3, 0:3] = torch.einsum('bxy,bt -> bxy', dr.Log().Jr(), dt)

        Ba = torch.zeros(B, 9, 3, device=dt.device, dtype=dt.dtype)
        Ba[..., 3:6, 0:3] = torch.einsum('bxy,bt -> bxy', incre_r.matrix(), dt)
        Ba[..., 6:9, 0:3] = 0.5 * torch.einsum('bxy,bt -> bxy', dr.matrix(), dt**2)

        B =  torch.einsum('bxy,bt -> bxy', Bg @ Cg @ Bg.mT + Ba @ Ca @ Ba.mT, 1 / dt)
        cov = A @ init_cov @ A.mT + B

        return {'cov':cov, 'A':A, 'Ba':Ba, 'Bg':Bg}

    @classmethod
    def batch_imu_forward(self, init_state, incre_state):
        return {
            'rot': init_state['r'] * incre_state["incre_r"],
            'vel': init_state['v'] + init_state['r'] * incre_state["incre_v"],
            'pos': init_state['p'] + init_state['r'] * incre_state["incre_p"] + init_state['v'] * incre_state["incre_t"],
        }

    @classmethod
    def batch_imu_integrate(self, init_state, dt, ang, acc, rot:pp.SO3=None, cov=None, gravity=9.81007, cov_propogation = True):
        """
        Args:
            acc: (B, Frames, 3)
            ang: (B, Frames, 3)
            dt: (B, Frames, 1)
            rot: (B, Frames, 4)
            init_state: dict

        dr, dp, dv: represents the incremental of one small interval 
        incre_r, incre_p, incre_v: represents the incremental starting from the first frame
        """
        assert(len(acc.shape) == 3)
        assert(len(dt.shape) == 3)
        assert(len(ang.shape) == 3)

        if rot is not None:
            assert(len(rot.shape) == 3)
        
        incre_state = self.batch_imu_update(init_state, dt, ang, acc, rot, gravity = gravity)
        inte_state = self.batch_imu_forward(init_state, incre_state)

        if cov_propogation:
            cov_state = self.cum_cov_propopate(dt = dt, acc = acc, incre_state=incre_state, init_cov = cov)
        else:
            cov_state = None

        return inte_state, cov_state

    @classmethod
    def cum_cov_propopate(self, dt, acc, incre_state, init_cov=None, ang_cov=(1.6968*10**-4)**2, acc_cov=(2.0*10**-3)**2):
        """
            Covariance propogation
            \Sigma_i^k
            
        """

        B, F = dt.shape[:2]
        Cg = torch.eye(3, device=dt.device, dtype=dt.dtype) * ang_cov
        Cg = Cg.repeat([B, F, 1, 1])
        Ca = torch.eye(3, device=dt.device, dtype=dt.dtype) * acc_cov
        Ca = Ca.repeat([B, F, 1, 1])


        if init_cov is None:
            init_cov = torch.zeros((B,9,9), device=dt.device, dtype=dt.dtype)

        Ha = pp.vec2skew(acc)

        A = torch.eye(9, device=dt.device, dtype=dt.dtype).repeat([B, F+1, 1, 1])
        A[:, :-1, 0:3, 0:3] = incre_state["dr"].matrix().mT
        A[:, :-1, 3:6, 0:3] = torch.einsum('...xy,...t -> ...xy', - incre_state["incre_r"].matrix() @ Ha, dt)
        A[:, :-1, 6:9, 0:3] = torch.einsum('...xy,...t -> ...xy', - 0.5 * incre_state["incre_r"].matrix() @ Ha, dt**2)
        A[:, :-1, 6:9, 3:6] = torch.einsum('...xy,...t -> ...xy', torch.eye(3, device=dt.device, dtype=dt.dtype).repeat([B, F, 1, 1]), dt)

        Bg = torch.zeros(B, F, 9, 3, device=dt.device, dtype=dt.dtype)
        Bg[..., 0:3, 0:3] = torch.einsum('...xy,...t -> ...xy', incre_state["dr"].Log().Jr(), dt)

        Ba = torch.zeros(B, F, 9, 3, device=dt.device, dtype=dt.dtype)
        Ba[..., 3:6, 0:3] = torch.einsum('...xy,...t -> ...xy', incre_state["incre_r"].matrix(), dt)
        Ba[..., 6:9, 0:3] = 0.5 * torch.einsum('...xy,...t -> ...xy', incre_state["dr"].matrix(), dt**2)

        # the term of B
        B_cov = torch.einsum('...xy,...t -> ...xy', Bg @ Cg @ Bg.mT + Ba @ Ca @ Ba.mT, 1/dt)
        B_cov = torch.cat([init_cov[:,None,...], B_cov], dim=1)

        A_left_cum = pp.cumprod(A.flip([1]), dim=1).flip([1]) # cumfrom An to I, then flip
        A_right_cum = A_left_cum.mT
        
        cov = (A_left_cum @ B_cov @ A_right_cum).sum(dim=1)

        return {"cov": cov, "A": A, "Ba": Ba, "Bg": Bg}
