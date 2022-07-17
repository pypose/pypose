import torch
import pypose as pp
from torch import nn


class IMUPreintegrator(nn.Module):
    r'''
    Applies preintegration over IMU input signals.

    Args:
        pos (torch.Tensor, optional): initial position. Default: torch.zeros(3)
        rot (pypose.SO3, optional): initial rotation. Default: :meth:`pypose.identity_SO3`
        vel (torch.Tensor, optional): initial position. Default: torch.zeros(3)
        gravity (float, optional): the gravity acceleration. Default: 9.81007
        gyro_cov (float, optional): covariance of the gyroscope. Default: (1.6968e-4)**2
        acc_cov (float, optional): covariance of the accelerator. Default: (2e-3)**2
        prop_cov (Bool, optional): flag to propagate the covariance matrix. Default: :obj:`True`
        reset (Bool, optional): flag to reset the initial states after each time the :obj:`forward`
            function is called. If False, the IMU integrator will use the states from last time
            as the initial states. Note that if the :obj:`init_state` is not :obj:`None` while calling 
            the :obj:`forward` function, the integrator will use the given initial state Default: :obj:`False`.
    '''
    def __init__(self, pos = torch.zeros(3),
                       rot = pp.identity_SO3(),
                       vel = torch.zeros(3),
                       gravity = 9.81007,
                       gyro_cov = (1.6968e-4)**2,
                       acc_cov = (2e-3)**2,
                       prop_cov = True,
                       reset = False):
        super().__init__()
        self.reset, self.prop_cov, self.gyro_cov, self.acc_cov = reset, prop_cov, gyro_cov, acc_cov
        # Initial status of IMU: (pos)ition, (rot)ation, (vel)ocity, (cov)ariance
        self.register_buffer('gravity', torch.tensor([0, 0, gravity]), persistent=False)
        self.register_buffer('pos', self._check(pos).clone(), persistent=False)
        self.register_buffer('rot', self._check(rot).clone(), persistent=False)
        self.register_buffer('vel', self._check(vel).clone(), persistent=False)
        self.register_buffer('cov', torch.zeros(1, 9, 9), persistent=False)

    def _check(self, obj):
        if obj is not None:
            if len(obj.shape) == 2:
                obj = obj[None, ...]

            elif len(obj.shape) == 1:
                obj = obj[None, None, ...]
        return obj

    def forward(self, dt, gyro, acc, rot:pp.SO3=None, gyro_cov=None, acc_cov=None, init_state=None):
        r"""
        Propagate IMU states from duration (:math:`\delta t`), gyroscope (angular rate :math:`\omega`),
        linear acceleration (:math:`\mathbf{a}`) in body frame, as well as their measurement
        covariance for gyroscope :math:`C_{g}` and acceleration :math:`C_{\mathbf{a}}`.
        Known IMU rotation :math:`R` estimation can also be provided for better precision.

        Args:
            dt (torch.Tensor): time interval from last update.
            gyro (torch.Tensor): angular rate (:math:`\omega`) in IMU body frame.
            acc (torch.Tensor): linear acceleration (:math:`\mathbf{a}`) in IMU body frame.
            rot (:obj:`pypose.SO3`, optional): known IMU rotation.
            gyro_cov (torch.Tensor, optional): covariance matrix of angular rate.
                Default value is used if not given.
            acc_cov (torch.Tensor, optional): covariance matrix of linear acceleration.
                Default value is used if not given.
            init_state (Dict, optional): the initial state for the integration. The structure
                of the dictionary should be :obj:`{'pos': torch.Tensor, 'rot': pypose.SO3, 'vel':
                torch.Tensor}`. The initial state given in constructor will be used if not given.

        Note:
            This layer supports the input shape with :math:`(B, F, H_{in})`, :math:`(F, H_{in})`
            and :math:`(H_{in})`, where :math:`B` is the batch size (or the number of IMU),
            :math:`F` is the number of frames (measurements), and :math:`H_{in}` is the raw
            sensor inputs.

        IMU Measurements Integration:

        .. math::
            \begin{align*}
                {\Delta}R_{ik+1} &= {\Delta}R_{ik} \mathrm{Exp} ((w_k - b_i^g) {\Delta}t) \\
                {\Delta}v_{ik+1} &= {\Delta}v_{ik} + {\Delta}R_{ik} (a_k - b_i^a) {\Delta}t \\
                {\Delta}p_{ik+1} &= {\Delta}v_{ik} + {\Delta}v_{ik} {\Delta}t
                    + 1/2 {\Delta}R_{ik} (a_k - b_i^a) {\Delta}t^2
            \end{align*}

        where:

            - :math:`{\Delta}R_{ik}` is the preintegrated rotation between the :math:`i`-th
              and :math:`k`-th time step.

            - :math:`{\Delta}v_{ik}` is the preintegrated velocity between the :math:`i`-th
              and :math:`k`-th time step.

            - :math:`{\Delta}p_{ik}` is the preintegrated position between the :math:`i`-th
              and :math:`k`-th time step.

            - :math:`a_k` is linear acceleration at the :math:`k`-th time step.

            - :math:`w_k` is angular rate at the :math:`k`-{th} time step.

        Uncertainty Propagation:

        .. math::
            \begin{align*}
                C_{ik+1} &= A C_{ik} A^T + B \mathrm{diag}(C_g, C_a) B^T \\
                  &= A C A^T + B_g C_g B_g^T + B_a C_a B_a^T
            \end{align*},

        where

        .. math::
            A = \begin{bmatrix}
                  {\Delta}R_{ik+1}^T & 0_{3*3} \\
                  -{\Delta}R_{ik} (a_k - b_i^g)^\wedge {\Delta}t & I_{3*3} & 0_{3*3} \\
                  -1/2{\Delta}R_{ik} (a_k - b_i^g)^\wedge {\Delta}t^2 & I_{3*3} {\Delta}t & I_{3*3}
                \end{bmatrix},

        .. math::
            B = [B_g, B_a] \\

        .. math::
            B_g = \begin{bmatrix}
                    J_r^k \Delta t  \\
                    0_{3*3}  \\
                    0_{3*3} 
                \end{bmatrix},

            B_a = \begin{bmatrix}
                    0_{3*3} \\
                    {\Delta}R_{ik} {\Delta}t  \\
                    1/2 {\Delta}R_{ik} {\Delta}t^2
                \end{bmatrix},

        where :math:`\cdot^\wedge` is the skew matrix (:meth:`pypose.vec2skew`),
        :math:`C \in\mathbf{R}^{9\times 9}` is the covariance matrix,
        and :math:`J_r^k` is the right jacobian (:meth:`pypose.Jr`) of integrated rotation
        :math:`\mathrm{Exp}(w_k{\Delta}t)` at :math:`k`-th time step,
        :math:`C_{g}` and :math:`C_{\mathbf{a}}` are measurement covariance of angular rate
        and acceleration, respectively.

        Note:
            Output covariance (Shape: (9, 9)) is in the order of rotation, velocity, and position.

        With IMU preintegration, the propagated IMU status:

        .. math::
            \begin{align*}
                R_j &= {\Delta}R_{ij} * R_i                                                     \\
                v_j &= {\Delta}v_{ij} * R_i   + v_i + g \Delta t_{ij}                           \\
                p_j &= {\Delta}p_{ij} * R_i   + p_i + v_i \Delta t_{ij} + 1/2 g \Delta t_{ij}^2 \\
            \end{align*}

        where:

            - :math:`{\Delta}R_{ij}`, :math:`{\Delta}v_{ij}`, :math:`{\Delta}p_{ij}`
              are the preintegrated measurements.
            - :math:`R_i`, :math:`v_i`, and :math:`p_i` are the initial state. Default initial values
              are used if :obj:`reset` is True.
            - :math:`R_j`, :math:`v_j`, and :math:`p_j` are the propagated state variables

        Note:
            The implementation is based on Eq. (A7), (A8), (A9), and (A10) of this report:

            * Christian Forster, et al., `IMU Preintegration on Manifold for Efficient Visual-Inertial
              Maximum-a-Posteriori Estimation
              <https://rpg.ifi.uzh.ch/docs/RSS15_Forster_Supplementary.pdf>`_, Technical Report
              GT-IRIM-CP&R-2015-001, 2015.

        Example:

            1. Preintegrator Initialization

            >>> import torch
            >>> import pypose as pp
            >>> p = torch.zeros(3)    # Initial Position
            >>> r = pp.identity_SO3() # Initial rotation
            >>> v = torch.zeros(3)    # Initial Velocity
            >>> integrator = pp.module.IMUPreintegrator(p, r, v)

            2. Get IMU measurement

            >>> ang = torch.tensor([0.1,0.1,0.1]) # angular velocity 
            >>> acc = torch.tensor([0.1,0.1,0.1]) # acceleration
            >>> rot = pp.mat2SO3(torch.eye(3))    # Rotation (Optional)
            >>> dt = torch.tensor([0.002])        # Time difference between two measurements

            3. Preintegrating IMU measurements.
            Takes as input the IMU values and calculates the preintegrated IMU measurements.

            >>> states = integrator(dt, ang, acc, rot)
            {'rot': SO3Type LieTensor:
            tensor([[[1.0000e-04, 1.0000e-04, 1.0000e-04, 1.0000e+00]]]),
            'vel': tensor([[[ 0.0002,  0.0002, -0.0194]]]),
            'pos': tensor([[[ 2.0000e-07,  2.0000e-07, -1.9420e-05]]]),
            'cov': tensor([[[ 5.7583e-11, -5.6826e-19, -5.6827e-19,  0.0000e+00,  0.0000e+00,
                              0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
                            [-5.6826e-19,  5.7583e-11, -5.6827e-19,  0.0000e+00,  0.0000e+00,
                              0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
                            [-5.6827e-19, -5.6827e-19,  5.7583e-11,  0.0000e+00,  0.0000e+00,
                              0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
                            [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  8.0000e-09, -3.3346e-20,
                             -1.0588e-19,  8.0000e-12,  1.5424e-23, -1.0340e-22],
                            [ 0.0000e+00,  0.0000e+00,  0.0000e+00, -1.3922e-19,  8.0000e-09,
                              0.0000e+00, -8.7974e-23,  8.0000e-12,  0.0000e+00],
                            [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00, -1.0588e-19,
                              8.0000e-09,  0.0000e+00, -1.0340e-22,  8.0000e-12],
                            [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  8.0000e-12,  1.5424e-23,
                             -1.0340e-22,  8.0000e-15, -1.2868e-26,  0.0000e+00],
                            [ 0.0000e+00,  0.0000e+00,  0.0000e+00, -8.7974e-23,  8.0000e-12,
                              0.0000e+00, -1.2868e-26,  8.0000e-15,  0.0000e+00],
                            [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00, -1.0340e-22,
                              8.0000e-12,  0.0000e+00,  0.0000e+00,  8.0000e-15]]])}

        Preintegrated IMU odometry from the KITTI dataset with and without known rotation.

            .. list-table:: 

                * - .. figure:: /_static/img/module/imu/imu-known-rot.png
                        :width: 300

                    Fig. 1. Known Rotation.

                  - .. figure:: /_static/img/module/imu/imu-unknown-rot.png
                        :width: 300

                    Fig. 2. Estimated Rotation.
        """
        assert(0 < len(acc.shape) == len(dt.shape) == len(gyro.shape) <= 3)
        acc = self._check(acc); gyro = self._check(gyro)
        dt = self._check(dt); rot = self._check(rot)

        if init_state is None:
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
            cov = self.propagate_cov(integrate, init_cov, gyro_cov, acc_cov)
        else:
            cov = {'cov': None}

        if not self.reset:
            self.pos = predict['pos'][..., -1:, :]
            self.rot = predict['rot'][..., -1:, :]
            self.vel = predict['vel'][..., -1:, :]
            self.cov = cov['cov']

        return {**predict, **cov}

    def integrate(self, init_state, dt, gyro, acc, rot:pp.SO3=None):
        B, F = dt.shape[:2]
        dr = torch.cat([pp.identity_SO3(B, 1, dtype=dt.dtype, device=dt.device), pp.so3(gyro*dt).Exp()], dim=1)
        incre_r = pp.cumprod(dr, dim = 1, left=False)
        inte_rot = init_state['rot'] * incre_r

        if isinstance(rot, pp.LieTensor):
            a = acc - rot.Inv() @ self.gravity
        else:
            a = acc - inte_rot[:,1:,:].Inv() @ self.gravity

        dv = torch.zeros(B, 1, 3, dtype=dt.dtype, device=dt.device)
        dv = torch.cat([dv, incre_r[:,:F,:] @ a * dt], dim=1)
        incre_v = torch.cumsum(dv, dim =1)

        dp = torch.zeros(B, 1, 3, dtype=dt.dtype, device=dt.device)
        dp = torch.cat([dp, incre_v[:,:F,:] * dt + incre_r[:,:F,:] @ a * 0.5 * dt**2], dim =1)
        incre_p = torch.cumsum(dp, dim =1)

        incre_t = torch.cumsum(dt, dim = 1)
        incre_t = torch.cat([torch.zeros(B, 1, 1, dtype=dt.dtype, device=dt.device), incre_t], dim =1)

        return {'acc':acc, 'vel':incre_v[...,1:,:], 'pos':incre_p[:,1:,:], 'rot':incre_r[:,1:,:],
                't':incre_t[...,1:,:], 'dr': dr[:,1:,:], 'dt': dt}

    @classmethod
    def predict(cls, init_state, integrate):
        return {
            'rot': init_state['rot'] * integrate['rot'],
            'vel': init_state['vel'] + init_state['rot'] * integrate['vel'],
            'pos': init_state['pos'] + init_state['rot'] * integrate['pos'] + init_state['vel'] * integrate['t'],
        }

    @classmethod
    def propagate_cov(cls, integrate, init_cov, gyro_cov, acc_cov):
        B, F = integrate['dt'].shape[:2]
        device = integrate['dt'].device; dtype = integrate['dt'].dtype
        Cg = torch.eye(3, device=device, dtype=dtype) * gyro_cov
        Cg = Cg.repeat([B, F, 1, 1])
        Ca = torch.eye(3, device=device, dtype=dtype) * acc_cov
        Ca = Ca.repeat([B, F, 1, 1])

        Ha = pp.vec2skew(integrate['acc'])
        A = torch.eye(9, device=device, dtype=dtype).repeat([B, F+1, 1, 1])
        A[:, :-1, 0:3, 0:3] = integrate['dr'].matrix().mT
        A[:, :-1, 3:6, 0:3] = torch.einsum('...xy,...t -> ...xy', \
            - integrate['rot'].matrix() @ Ha, integrate['dt'])
        A[:, :-1, 6:9, 0:3] = torch.einsum('...xy,...t -> ...xy', \
            - 0.5 * integrate['rot'].matrix() @ Ha, integrate['dt']**2)
        A[:, :-1, 6:9, 3:6] = torch.einsum('...xy,...t -> ...xy', \
            torch.eye(3, device=device, dtype=dtype).repeat([B, F, 1, 1]), integrate['dt'])

        Bg = torch.zeros(B, F, 9, 3, device=device, dtype=dtype)
        Bg[..., 0:3, 0:3] = torch.einsum('...xy,...t -> ...xy', integrate['dr'].Log().Jr(), integrate['dt'])

        Ba = torch.zeros(B, F, 9, 3, device=device, dtype=dtype)
        Ba[..., 3:6, 0:3] = torch.einsum('...xy,...t -> ...xy', integrate['rot'].matrix(), integrate['dt'])
        Ba[..., 6:9, 0:3] = 0.5 * torch.einsum('...xy,...t -> ...xy', integrate['dr'].matrix(), integrate['dt']**2)

        # the term of B
        init_cov = init_cov.expand(B, 9, 9)
        B_cov = torch.einsum('...xy,...t -> ...xy', Bg @ Cg @ Bg.mT + Ba @ Ca @ Ba.mT, 1/integrate['dt'])
        B_cov = torch.cat([init_cov[:,None,...], B_cov], dim=1)

        A_left_cum = pp.cumprod(A.flip([1]), dim=1).flip([1]) # cumfrom An to I, then flip
        A_right_cum = A_left_cum.mT

        cov = (A_left_cum @ B_cov @ A_right_cum).sum(dim=1)

        return {'cov': cov}
