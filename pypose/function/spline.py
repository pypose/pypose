import torch
from . import is_SE3


def CSplineR3(points, steps=0.1):
    r"""
    Cubic Hermite Spline in R3

    Args:
        points (:obj:`Tensor`): the sparse points for interpolation with
            [batch_size, point_num, 3] shape.
        steps (:obj:`Float`): the step between adjacant interpolation points.

    Returns:
       ``Tensor``: the interpolated points with [batch_size, inter_points_num, 3] shape.

    A points query at


    - .. figure:: /_static/img/module/liespline/CsplineR3.png
        :width: 600

        Fig. 1. Result of Cubic Spline Interpolation in R3.
    """
    batch_size, num_p, _ = points.shape
    xs = torch.arange(0, num_p-1+steps, steps, device=points.device)
    xs = xs.repeat(batch_size, 1)
    x  = torch.arange(num_p, device=points.device, dtype=points.dtype)
    x  = x.repeat(batch_size, 1)
    m = (points[..., 1:, :] - points[..., :-1, :])
    m /= (x[..., 1:] - x[..., :-1]).unsqueeze(2)
    m = torch.cat([m[...,[0],:], (m[..., 1:,:] + m[..., :-1,:]) / 2, m[...,[-1],:]], 1)
    idxs = torch.searchsorted(x[0, 1:], xs[0, :])
    dx = x[..., idxs + 1] - x[..., idxs]
    t = (xs - x[:, idxs]) / dx
    alpha = torch.arange(4, device=t.device, dtype=t.dtype)
    tt = t[:, None, :]**alpha[None, :, None]
    A = torch.tensor([[1, 0, -3, 2],
                      [0, 1, -2, 1],
                      [0, 0, 3, -2],
                      [0, 0, -1, 1]], dtype=t.dtype, device=t.device)
    hh = A@tt
    hh = torch.transpose(hh, 1, 2)
    out = hh[..., 0:1] * points[..., idxs, :]
    out = out + hh[..., 1:2] * m[..., idxs, :] * dx[..., None]
    out = out + hh[..., 2:3] * points[:, idxs + 1, :]
    out = out + hh[..., 3:4] * m[..., idxs + 1, :] * dx[..., None]
    return out

def BSplineSE3(input_poses, time):
    r'''
    B-spline interpolation in SE3.

    Args:
        input_poses (:obj:`LieTensor`): the input sparse poses with
            [batch_size, point_num, 7] shape.
        time (:obj:`Tensor`): the k time point with [1, 1, k] shape.

    Returns:
        ``LieTensor``: the interpolated m poses with [batch_size, m, 7]

    A poses query at any time :math:`t \in [t_i, t_{i+1})` (i.e. a segment of the spline)
    only relies on the poses located at times :math:`\{t_{i-1},t_i,t_{i+1},t_{i+2}\}`.
    It means that the interpolation between adjacent poses needs four consecutive poses.
    Thus, the B-spline interpolation could estimate the pose between

    :math:`[t_1, t_{n-1}]` by the input poses at :math:`\{t_0, ...,t_{n}\}`.

    The absolute pose of the spline :math:`T_{s}^w(t)`, where :math:`w` denotes the world
    and :math:`s` is the spline coordinate frame, can be calculated:

    .. math::
        \begin{aligned}
            &T^w_s(t) = T^w_{i-1} \prod^{i+2}_{j=i}\delta T_j,\\
            &\delta T_j= \exp(\lambda_{j-i}(t)\delta \hat{\xi}_j),\\
            &\lambda(t) = MU(t),
        \end{aligned}

    :math:`M` is a predefined matrix, shown as follows:

    .. math::
        M = \frac{1}{6}\begin{bmatrix}
            5& 3& -3& 1 \\
            1& 3& 3& -2\\
            0& 0& 0& 1 \\
        \end{bmatrix}

    :math:`U(t)` is a vector:

    .. math::
        U(t) = \begin{bmatrix}
                1\\
                u(t)\\
                u(t)^2\\
                u(t)^3\\
            \end{bmatrix}

    where :math:`u(t)=(t-t_i)/\Delta t`, :math:`\Delta t = t_{i+1} - t_{i}`. :math:`t_i`
    is the time point of the :math:`i_{th}` given pose. :math:`t` is the interpolation
    time point :math:`t \in [t_{i},t_{i+1})`.

    :math:`\delta \hat{\xi}_j` is the transformation between :math:`\hat{\xi}_{j-1}`
    and :math:`\hat{\xi}_{j}`

    .. math::
        \begin{aligned}
            \delta \hat{\xi}_j :&= \log(\exp(\hat{\xi} _w^{j-1})\exp(\hat{\xi}_j^w))
                                &= \log(T_j^{j-1})\in \mathfrak{se3}
        \end{aligned}

    Note:
        The implementation is based on Eq. (A7), (A8), (A9), and (A10) of this report:
        * David Hug, et al.,
        `HyperSLAM: A Generic and Modular Approach to Sensor Fusion and Simultaneous
        Localization And Mapping in Continuous-Time
        <https://ieeexplore.ieee.org/abstract/document/9320417>`_,
        2020 International Conference on 3D Vision (3DV), Fukuoka, Japan, 2020.

    Examples:
        >>> import torch
        >>> import pypose as pp
        >>> a1 = pp.euler2SO3(torch.Tensor([0., 0., 0.]))
        >>> a2 = pp.euler2SO3(torch.Tensor([torch.pi / 4., torch.pi / 3., torch.pi / 2.]))
        >>> time = torch.arange(0, 1, 0.25).reshape(1, 1, -1)
        >>> poses = pp.LieTensor([[
        ...                             [0., 4., 0., a1[0], a1[1], a1[2], a1[3]],
        ...                             [0., 3., 0., a1[0], a1[1], a1[2], a1[3]],
        ...                             [0., 2., 0., a1[0], a1[1], a1[2], a1[3]],
        ...                             [0., 1., 0., a1[0], a1[1], a1[2], a1[3]],
        ...                             [1., 0., 1., a2[0], a2[1], a2[2], a2[3]],
        ...                             [2., 0., 1., a2[0], a2[1], a2[2], a2[3]],
        ...                             [3., 0., 1., a2[0], a2[1], a2[2], a2[3]],
        ...                             [4., 0., 1., a2[0], a2[1], a2[2], a2[3]]
        ...                            ]], ltype=pp.SE3_type)
        >>> wayposes = pp.BSplineSE3(poses, time)

    - .. figure:: /_static/img/module/liespline/BsplineSE3.png

        Fig. 1. Result of B Spline Interpolation in SE3.
    '''
    assert is_SE3(input_poses), "The input poses are not SE3Type."
    assert time.shape[0]==time.shape[1]==1, "The time has wrong shape."
    input_poses = torch.cat(
        [torch.cat([input_poses[..., [0], :], input_poses], dim=1),
            input_poses[..., [-1], :]], dim=1)
    timeSize = time.shape[-1]
    posesSize = input_poses.shape[1]
    alpha = torch.arange(4, dtype=time.dtype, device=time.device)
    tt = time[:, None, :] ** alpha[None, :, None]
    B = torch.tensor([[5, 3,-3, 1],
                      [1, 3, 3,-2],
                      [0, 0, 0, 1]], dtype=time.dtype, device=time.device) / 6
    w = (B @ tt).squeeze(0)
    w0 = w[..., 0, :].T
    w1 = w[..., 1, :].T
    w2 = w[..., 2, :].T
    posesTensor = torch.stack([input_poses[..., i:i + 4, :]
                                for i in range(posesSize - 3)], dim=1)
    T_delta = input_poses[..., 0:-3, :].unsqueeze(2).repeat(1, 1, timeSize, 1)
    A0 = ((posesTensor[..., [0], :].Inv() *
            (posesTensor[..., [1], :])).Log() * w0).Exp()
    A1 = ((posesTensor[..., [1], :].Inv() *
            (posesTensor[..., [2], :])).Log() * w1).Exp()
    A2 = ((posesTensor[..., [2], :].Inv() *
            (posesTensor[..., [3], :])).Log() * w2).Exp()
    interPosesSize = timeSize * (posesSize - 3)
    interPoses = (T_delta * A0 * A1 * A2).reshape((-1, interPosesSize, 7))
    return interPoses
