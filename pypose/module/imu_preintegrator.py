import torch
from torch import nn
from ..basics import cumprod
from .. import LieTensor, so3, SO3
from .. import identity_SO3, vec2skew


class IMUPreintegrator(nn.Module):
    r'''
    Applies preintegration over IMU input signals.

    Args:
        pos (``torch.Tensor``, optional): initial position. Default: :obj:`torch.zeros(3)`.
        rot (``pypose.SO3``, optional): initial rotation. Default: :meth:`pypose.identity_SO3`.
        vel (``torch.Tensor``, optional): initial position. Default: ``torch.zeros(3)``.
        gravity (``float``, optional): the gravity acceleration. Default: ``9.81007``.
        gyro_cov (``torch.Tensor`` or ``float``, optional): covariance of the gyroscope.
            Use a three-element tensor, if the covariance on the three axes are different.
            Default: ``(3.2e-3)**2``.
        acc_cov (``torch.Tensor``  or ``float``, optional): covariance of the accelerator.
            Use a three-element tensor, if the covariance on the three axes are different.
            Default: ``(8e-2)**2``.
        prop_cov (``bool``, optional): flag to propagate the covariance matrix. Default: ``True``.
        reset (``bool``, optional): flag to reset the initial states after the :obj:`forward`
            function is called. If ``False``, the integration starts from the last integration.
            This flag is ignored if :obj:`init_state` is not ``None``. Default: ``False``.

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

    Note:
        The examples generating the above figures can be found at `examples/module/imu
        <https://github.com/pypose/pypose/tree/main/examples/module/imu>`_.
    '''
    def __init__(self, pos = torch.zeros(3),
                       rot = identity_SO3(),
                       vel = torch.zeros(3),
                       gravity = 9.81007,
                       gyro_cov = (3.2e-3)**2,
                       acc_cov = (8e-2)**2,
                       prop_cov = True,
                       reset = False):
        super().__init__()
        self.reset, self.prop_cov = reset, prop_cov

        if isinstance(acc_cov, float):
            acc_cov = torch.tensor([[acc_cov, acc_cov, acc_cov]])
        if isinstance(gyro_cov, float):
            gyro_cov = torch.tensor([[gyro_cov, gyro_cov, gyro_cov]])
        
        # Initial status of IMU: (pos)ition, (rot)ation, (vel)ocity, (cov)ariance
        self.register_buffer('gravity', torch.tensor([0, 0, gravity]), persistent=False)
        self.register_buffer('pos', self._check(pos).clone(), persistent=False)
        self.register_buffer('rot', self._check(rot).clone(), persistent=False)
        self.register_buffer('vel', self._check(vel).clone(), persistent=False)
        self.register_buffer('cov', torch.zeros(1, 9, 9), persistent=False)
        self.register_buffer('gyro_cov', gyro_cov, persistent=False)
        self.register_buffer('acc_cov', acc_cov, persistent=False)
        self.Rij = None # rotation corresponding to the "zero-state" covariance Sigma_ii.

    def _check(self, obj):
        if obj is not None:
            if len(obj.shape) == 2:
                obj = obj[None, ...]

            elif len(obj.shape) == 1:
                obj = obj[None, None, ...]
        return obj

    def forward(self, dt, gyro, acc, rot:SO3=None, gyro_cov=None, acc_cov=None, init_state=None):
        r"""
        Propagate IMU states from duration (:math:`\delta t`), angular rate
        (:math:`\omega`), linear acceleration (:math:`\mathbf{a}`) in body frame, as well as
        their measurement covariance for gyroscope :math:`C_{g}` and acceleration
        :math:`C_{\mathbf{a}}`. Known IMU rotation estimation :math:`R` can be provided for
        better precision.

        Args:
            dt (``torch.Tensor``): time interval from last update.
            gyro (``torch.Tensor``): angular rate (:math:`\omega`) in IMU body frame.
            acc (``torch.Tensor``): linear acceleration (:math:`\mathbf{a}`) in IMU body frame
                (raw sensor input with gravity).
            rot (:obj:`pypose.SO3`, optional): known IMU rotation on the body frame.
            gyro_cov (``torch.Tensor``, optional): covariance matrix of angular rate.
                If not given, the default state in constructor will be used.
            acc_cov (``torch.Tensor``, optional): covariance matrix of linear acceleration.
                If not given, the default state in constructor will be used.
            init_state (``dict``, optional): the initial state of the integration. The dictionary
                should be in form of :obj:`{'pos': torch.Tensor, 'rot': pypose.SO3, 'vel':
                torch.Tensor}`. If not given, the initial state in constructor will be used.

        Shape:
            - input (:obj:`dt`, :obj:`gyro`, :obj:`acc`): This layer supports the input shape with
              :math:`(B, F, H_{in})`, :math:`(F, H_{in})` and :math:`(H_{in})`, where :math:`B` is
              the batch size (or the number of IMU), :math:`F` is the number of frames
              (measurements), and :math:`H_{in}` is the raw sensor signals.

            - init_state (Optional): The initial state of the integration. It contains
              :code:`pos`: initial position, :code:`rot`: initial rotation, :code:`vel`: initial
              velocity, with the shape :math:`(B, H_{in})`.

            - output: a :obj:`dict` of integrated state including ``pos``: position,
              ``rot``: rotation, and ``vel``: velocity, each of which has a shape
              :math:`(B, F, H_{out})`, where :math:`H_{out}` is the signal dimension.
              If :obj:`prop_cov` is ``True``, it will also include ``cov``: covariance
              matrix in shape of :math:`(B, 9, 9)`.

        IMU Measurements Integration:

        .. math::
            \begin{align*}
                {\Delta}R_{ik+1} &= {\Delta}R_{ik} \mathrm{Exp} (w_k {\Delta}t) \\
                {\Delta}v_{ik+1} &= {\Delta}v_{ik} + {\Delta}R_{ik} a_k {\Delta}t \\
                {\Delta}p_{ik+1} &= {\Delta}p_{ik} + {\Delta}v_{ik} {\Delta}t
                    + \frac{1}{2} {\Delta}R_{ik} a_k {\Delta}t^2
            \end{align*}

        where:

            - :math:`{\Delta}R_{ik}` is the preintegrated rotation between the :math:`i`-th
              and :math:`k`-th time step.

            - :math:`{\Delta}v_{ik}` is the preintegrated velocity between the :math:`i`-th
              and :math:`k`-th time step.

            - :math:`{\Delta}p_{ik}` is the preintegrated position between the :math:`i`-th
              and :math:`k`-th time step.

            - :math:`a_k` is linear acceleration at the :math:`k`-th time step.

            - :math:`w_k` is angular rate at the :math:`k`-th time step.

            - :math:`{\Delta}t` is the time interval from time step :math:`k`-th to time 
              step :math:`{k+1}`-th time step.

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
                  -{\Delta}R_{ik} (a_k^{\wedge}) {\Delta}t & I_{3*3} & 0_{3*3} \\
                  -1/2{\Delta}R_{ik} (a_k^{\wedge}) {\Delta}t^2 & I_{3*3} {\Delta}t & I_{3*3}
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
            Output covariance (Shape: :math:`(B, 9, 9)`) is in the order of rotation, velocity,
            and position.

        With IMU preintegration, the propagated IMU status:

        .. math::
            \begin{align*}
                R_j &= {\Delta}R_{ij} * R_i                                                     \\
                v_j &= R_i * {\Delta}v_{ij}   + v_i + g \Delta t_{ij}                           \\
                p_j &= R_i * {\Delta}p_{ij}   + p_i + v_i \Delta t_{ij} + 1/2 g \Delta t_{ij}^2 \\
            \end{align*}

        where:

            - :math:`{\Delta}R_{ij}`, :math:`{\Delta}v_{ij}`, :math:`{\Delta}p_{ij}`
              are the preintegrated measurements.
            - :math:`R_i`, :math:`v_i`, and :math:`p_i` are the initial state. Default initial values
              are used if :obj:`reset` is True.
            - :math:`R_j`, :math:`v_j`, and :math:`p_j` are the propagated state variables.
            - :math:`{\Delta}t_{ij}` is the time interval from frame i to j.

        Note:
            The implementation is based on Eq. (A7), (A8), (A9), and (A10) of this report:

            * Christian Forster, et al., `IMU Preintegration on Manifold for Efficient Visual-Inertial
              Maximum-a-posteriori Estimation
              <https://rpg.ifi.uzh.ch/docs/RSS15_Forster_Supplementary.pdf>`_, Technical Report
              GT-IRIM-CP&R-2015-001, 2015.
        """
        assert(0 < len(acc.shape) == len(dt.shape) == len(gyro.shape) <= 3)
        acc = self._check(acc); gyro = self._check(gyro)
        dt = self._check(dt); rot = self._check(rot)
        B = dt.shape[0]

        if init_state is None:
            init_state = {'pos': self.pos, 'rot': self.rot, 'vel': self.vel}

        inte_state = self.integrate(dt, gyro, acc, rot = rot, init_rot = init_state['rot'])
        predict = self.predict(init_state, inte_state)

        if self.prop_cov:
            if gyro_cov is None:
                gyro_cov = self.gyro_cov.repeat([B,1,1])
            if acc_cov is None:
                acc_cov = self.acc_cov.repeat([B,1,1])
            if 'cov' not in init_state or init_state['cov'] is None:
                init_cov = self.cov.expand(B, 9, 9)
            else:
                init_cov = init_state['cov']
            if 'Rij' in init_state:
                Rij = init_state['Rij']
            else:
                Rij = self.Rij # default is None

            if Rij is not None:
                Rij = Rij * inte_state['Dr']
            else:
                Rij = inte_state['Dr']

            cov_input_state ={
                'Rij': Rij.detach(),
                'Rk': inte_state['w'].detach(),
                'Ha': vec2skew(inte_state['a'].detach()),
                'dt': dt.detach() 
            }
            cov = self.propagate_cov(cov_input = cov_input_state, init_cov = init_cov,
                                    gyro_cov = gyro_cov, acc_cov = acc_cov)
        else:
            cov = {'cov': None}

        if not self.reset:
            self.pos = predict['pos'][..., -1:, :]
            self.rot = predict['rot'][..., -1:, :]
            self.vel = predict['vel'][..., -1:, :]
            self.cov = cov['cov']
            self.Rij = Rij[..., -1:, :]

        return {**predict, **cov}

    def integrate(self, dt, gyro, acc, rot:SO3=None, init_rot:SO3=None):
        r"""
        Integrate the IMU sensor signals gyroscope (angular rate
        :math:`\omega`), linear acceleration (:math:`\mathbf{a}`) in body frame to
        calculate the increments on the rotation (:math:`\Delta r`), velocity (:math:`\Delta v`)
        and position (:math:`\Delta p`) of the IMU states.
        The IMU rotation of the body frame (:code:`\rot`) is optional,
        which can be utilized to compensate the gravity.

        .. math::
            \begin{align*}
                {\Delta}r_{ij} &= \int_{t \in [t_i, t_j]} w_k \ dt \\
                {\Delta}v_{ij} &= \int_{t \in [t_i, t_j]} R_{k} a_k \ dt \\
                {\Delta}p_{ij} &= \int\int_{t \in [t_i, t_j]} R_{k} a_k\ dt^2\  \\
            \end{align*}

        Args:
            dt (``torch.Tensor``): time interval from last update.
            gyro (torch.Tensor): angular rate (:math:`\omega`) in IMU body frame.
            acc (``torch.Tensor``): linear acceleration (:math:`\mathbf{a}`) in IMU body frame
                (raw sensor input with gravity).
            rot (:obj:`pypose.SO3`, optional): known IMU rotation on the body frame.
            init_rot (:obj:`pypose.SO3`, optional): the initial orientation of the IMU state.
                If not given, the initial state in constructor will be used.

        Shape:
            - input (:obj:`dt`, :obj:`gyro`, :obj:`acc`): This layer supports the input shape
              with :math:`(B, F, H_{in})`, :math:`(F, H_{in})` and :math:`(H_{in})`, where
              :math:`B` is the batch size (or the number of IMU), :math:`F` is the number of
              frames (measurements), and :math:`H_{in}` is the raw sensor signals.

            - init_rot: The initial orientation of the integration, which helps to 
              compensate for the gravity. It contains the shape :math:`(B, H_{in})`. d

            - rot: The ground truth orientation of the integration. If this parameter is
              given, the integrator will use the ground truth orientation to compensate the
              gravity.

        Return:
            ``dict``: integrated states including ``a``: acceleration in the body frame
            without gravity ``Dp``: position increments, ``Dr``: rotation increments,
            ``Dv``: velocity increments, ``w``: rotation velocity and ``Dt``: time increments,
            each of which has a shape :math:`(B, F, H_{out})`, where :math:`H_{out}` is the
            signal dimension.
        """
        B, F = dt.shape[:2]
        dr =  so3(gyro*dt).Exp()
        w = torch.cat([identity_SO3(B, 1, dtype=dt.dtype, device=dt.device), dr], dim=1)
        incre_r = cumprod(w, dim = 1, left=False)

        if isinstance(rot, LieTensor):
            a = acc - rot.Inv() @ self.gravity
        else:
            if init_rot is None:
                init_rot = identity_SO3(B, 1, dtype=dt.dtype, device=dt.device)
            inte_rot = init_rot * incre_r
            a = acc - inte_rot[:,1:,:].Inv() @ self.gravity

        dv = torch.zeros(B, 1, 3, dtype=dt.dtype, device=dt.device)
        dv = torch.cat([dv, incre_r[:,:F,:] @ a * dt], dim=1)
        incre_v = torch.cumsum(dv, dim =1)

        dp = torch.zeros(B, 1, 3, dtype=dt.dtype, device=dt.device)
        dp = torch.cat([dp, incre_v[:,:F,:] * dt + incre_r[:,:F,:] @ a * 0.5 * dt**2], dim =1)
        incre_p = torch.cumsum(dp, dim =1)

        incre_t = torch.cumsum(dt, dim = 1)
        incre_t = torch.cat([torch.zeros(B, 1, 1, dtype=dt.dtype, device=dt.device), incre_t], dim =1)

        return {'a':a, 'Dp':incre_p[:,1:,:], 'Dv':incre_v[...,1:,:], 'Dr':incre_r[:,1:,:],
                'Dt':incre_t[...,1:,:], 'w': w[:,1:,:]}

    @classmethod
    def predict(cls, init_state, integrate):
        r"""
        Propogate the next IMU state from the initial IMU state (:obj:`init_state`) with 
        the preintegrated IMU measurements (:math:`\Delta{p}`, :math:`\Delta{v}` and
        :math:`\Delta{r}`).

        .. math::
            \begin{align*}
                R_j &= {\Delta}R_{ij} * R_i                                                     \\
                v_j &= R_i * {\Delta}v_{ij}   + v_i + g \Delta t_{ij}                           \\
                p_j &= R_i * {\Delta}p_{ij}   + p_i + v_i \Delta t_{ij} + 1/2 g \Delta t_{ij}^2 \\
            \end{align*}

        Args:
            init_state (``dict``): the initial state of the integration. The dictionary
                should be in form of :obj:`{'pos': torch.Tensor, 'rot': pypose.SO3, 'vel':
                torch.Tensor}`.
            integrate (``dict``): the preintegrated IMU measurements. The dictionary
                should be in form of :obj:`{'Dp': torch.Tensor, 'Dr': pypose.SO3, 'Dv':
                torch.Tensor, 'Dt': torch.Tensor}`.

        Shape:
            - init_state: The initial state of the integration. It contains :code:`pos`: initial
              position, :code:`rot`: initial rotation, :code:`vel`: initial velocity, with the
              shape :math:`(B, H_{in})`.

            - integrate: The preintegrated IMU measurements. It contains :obj:`Dp`, :obj:`Dv`,
              :obj:`Dr`, and :obj:`Dt`, with the shape :math:`(B, F, H_{out})`. It follows the
              output of the function :obj:`integrate`.

        Return:
            ``dict``: integrated states including ``pos``: position, ``rot``: rotation, and
            ``vel``: velocity, each of which has a shape :math:`(B, F, H_{out})`, where
            :math:`H_{out}` is the signal dimension.
        """
        return {
            'rot': init_state['rot'] * integrate['Dr'],
            'vel': init_state['vel'] + init_state['rot'] * integrate['Dv'],
            'pos': init_state['pos'] + init_state['rot'] * integrate['Dp'] + init_state['vel'] * integrate['Dt'],
        }

    @classmethod
    def propagate_cov(cls, cov_input, init_cov, gyro_cov, acc_cov):
        ## The input acc_cov and gyro cov should be a vector of 3
        B, F = cov_input['dt'].shape[:2]
        device = cov_input['dt'].device; dtype = cov_input['dt'].dtype

        Cg = torch.diag_embed(gyro_cov)
        Ca = torch.diag_embed(acc_cov)

        # constructing the propagate 
        A = torch.eye(9, device=device, dtype=dtype).repeat([B, F+1, 1, 1])
        Bg = torch.zeros(B, F, 9, 3, device=device, dtype=dtype)
        Ba = torch.zeros(B, F, 9, 3, device=device, dtype=dtype)
        
        A[:, :-1, 0:3, 0:3] = cov_input['Rk'].matrix().mT # R_{k,k+1}^T
        A[:, :-1, 3:6, 0:3] = torch.einsum('...xy,...t -> ...xy', \
            - cov_input['Rij'].matrix() @ cov_input['Ha'], cov_input['dt'])
        A[:, :-1, 6:9, 0:3] = torch.einsum('...xy,...t -> ...xy', \
            - 0.5 * cov_input['Rij'].matrix() @ cov_input['Ha'], cov_input['dt']**2)
        A[:, :-1, 6:9, 3:6] = torch.einsum('...xy,...t -> ...xy', \
            torch.eye(3, device=device, dtype=dtype).repeat([B, F, 1, 1]), cov_input['dt'])

        # [J dt, 0, 0]^T
        Bg[..., 0:3, 0:3] = torch.einsum('...xy,...t -> ...xy', cov_input['Rk'].Jr(), cov_input['dt'])

        # [0, R_{ik}dt, 1/2R_{ik}dt^2]^T
        Ba[..., 3:6, 0:3] = torch.einsum('...xy,...t -> ...xy', cov_input['Rij'].matrix(), cov_input['dt'])
        Ba[..., 6:9, 0:3] = 0.5 * torch.einsum('...xy,...t -> ...xy',
                                                cov_input['Rij'].matrix(), cov_input['dt']**2)

        # the term of B
        B_cov = torch.einsum('...xy,...t -> ...xy', Bg @ Cg @ Bg.mT + Ba @ Ca @ Ba.mT, 1/cov_input['dt'])
        B_cov = torch.cat([init_cov[:,None,...], B_cov], dim=1)

        A_left_cum = cumprod(A.flip([1]), dim=1).flip([1]) # cum from An to I, then flip
        A_right_cum = A_left_cum.mT
        cov = torch.sum(A_left_cum @ B_cov @ A_right_cum, dim=1)
        return {'cov': cov, 'Rij': cov_input['Rij'][..., -1:, :]}
