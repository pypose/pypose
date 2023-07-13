import torch
from . import is_SE3


def chspline(points, interval=0.1):
    r"""
    Cubic Hermite Spline, a piecewise-cubic interpolator matching values and first
    derivatives.

    Args:
        points (:obj:`Tensor`): the sequence of points for interpolation with shape
            [..., point_num, dim].
        interval (:obj:`float`): the unit interval between interpolated points.
            We assume the interval of adjacent input points is 1. Therefore, if we set
            ``interval`` as ``0.1``, the interpolated points between the points at
            :math:`t` and :math:`t+1` will be :math:`[t, t+0.1, ..., t+0.9, t+1]`.
            Default: ``0.1``.

    Returns:
       ``Tensor``: the interpolated points.

    The interpolated points are evenly distributed between a start point and a end point
    according to the number of interpolated points. In this function, the interpolated points
    are also evenly distributed between the start point and end point.

    Denote the starting point :math:`p_0` with the starting tangent :math:`m_0` and the
    ending point :math:`p_1` with the ending tagent :math:`m_1`. The interpolated point at
    :math:`t` can be defined as:

    .. math::
        \begin{aligned}
            p(t) &= (1-3t^2+2t^3)p_0 + (t-2t^2+t^3)m_0+(3t^2-2t^3)p_1+(-t^2+t^3)m_1\\
                 &= \begin{bmatrix}
                        p_0, m_0, p_1, m_1
                    \end{bmatrix}
                    \begin{bmatrix}
                        1& 0&-3& 2\\
                        0& 1&-2& 1\\
                        0& 0& 3&-2\\
                        0& 0&-1& 1
                    \end{bmatrix}
                    \begin{bmatrix}1\\ t\\ t^2\\ t^3\end{bmatrix}

        \end{aligned}

    Note:
        The implementation is based on wiki of `Cubic Hermite spline
        <https://en.wikipedia.org/wiki/Cubic_Hermite_spline>`_.

    Examples:
        >>> import torch
        >>> import pypose as pp
        >>> points = torch.tensor([[[0., 0., 0.],
        ...                         [1., .5, 0.1],
        ...                         [0., 1., 0.2],
        ...                         [1., 1.5, 0.4],
        ...                         [1.5, 0., 0.],
        ...                         [2., 1.5, 0.4],
        ...                         [2.5, 0., 0.],
        ...                         [1.75, 0.75, 0.2],
        ...                         [2.25, 0.75, 0.2],
        ...                         [3., 1.5, 0.4],
        ...                         [3., 0., 0.],
        ...                         [4., 0., 0.],
        ...                         [4., 1.5, 0.4],
        ...                         [5., 1., 0.2],
        ...                         [4., 0.75, 0.2],
        ...                         [5., 0., 0.]]])
        >>> waypoints = pp.chspline(points, 0.1)

    .. figure:: /_static/img/module/liespline/CsplineR3.png
        :width: 600

        Fig. 1. Result of Cubic Spline Interpolation in 3D space.
    """
    assert points.dim() >= 2, "Dimension of points should be [..., N, C]"
    assert interval < 1.0, "The interval should be smaller than 1."
    batch, N = points.shape[:-2], points.shape[-2]
    dargs = {'device': points.device, 'dtype': points.dtype}
    intervals = torch.arange(0, 1, interval, **dargs)
    timeline = torch.arange(0, N, **dargs).unsqueeze(-1)
    timeline = (timeline + intervals).view(-1)[:-(intervals.shape[0]-1)]
    x = torch.arange(N, **dargs)
    idxs = torch.searchsorted(x[...,1:], timeline[..., :])
    x = x.expand(batch+(-1,))
    xs = timeline.expand(batch+(-1,))
    m = (points[..., 1:, :] - points[..., :-1, :])
    m /= (x[..., 1:] - x[..., :-1])[..., None]
    m = torch.cat([m[...,[0],:], (m[..., 1:,:] + m[..., :-1,:]) / 2, m[...,[-1],:]], -2)
    dx = x[..., idxs + 1] - x[..., idxs]
    t = (xs - x[..., idxs]) / dx
    alpha = torch.arange(4, **dargs)
    tt = t[..., None, :]**alpha[..., None]
    A = torch.tensor([[1, 0, -3, 2],
                      [0, 1, -2, 1],
                      [0, 0, 3, -2],
                      [0, 0, -1, 1]], **dargs)
    hh = (A @ tt).mT
    interpoints = hh[..., :1] * points[..., idxs, :]
    interpoints += hh[..., 1:2] * m[..., idxs, :] * dx[..., None]
    interpoints += hh[..., 2:3] * points[..., idxs + 1, :]
    interpoints += hh[..., 3:4] * m[..., idxs + 1, :] * dx[..., None]
    return interpoints

def bspline(data, interval=0.1, extrapolate=False):
    r'''
    B-spline interpolation, which currently only supports SE3 LieTensor.

    Args:
        data (:obj:`LieTensor`): the input sparse poses with
            [batch_size, poses_num, dim] shape.
        interval (:obj:`float`): the unit interval between interpolated poses.
            We assume the interval of adjacent input poses is 1. Therefore, if we set
            ``interval`` as ``0.1``, the interpolated poses between the poses at
            :math:`t` and :math:`t+1` will be at :math:`[t, t+0.1, ..., t+0.9, t+1]`.
            Default: ``0.1``.
        extrapolate(``bool``): flag to determine whether the interpolate poses pass the
            start and end poses. Default: ``False``.

    Returns:
        :obj:`LieTensor`: the interpolated SE3 LieTensor.

    A poses query at :math:`t \in [t_i, t_{i+1})` (i.e. a segment of the spline)
    only relies on the poses at four steps :math:`\{t_{i-1},t_i,t_{i+1},t_{i+2}\}`.
    It means that the interpolation between adjacent poses needs four consecutive poses.
    In this function, the interpolated poses are evenly distributed between a
    pose query according to the interval parameter.

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
        The implementation is based on Eq. (3), (4), (5), and (6) of this paper:
        David Hug, et al.,
        `HyperSLAM: A Generic and Modular Approach to Sensor Fusion and Simultaneous
        Localization And Mapping in Continuous-Time
        <https://ieeexplore.ieee.org/abstract/document/9320417>`_,
        2020 International Conference on 3D Vision (3DV), Fukuoka, Japan, 2020.

    Examples:
        >>> import torch, pypose as pp
        >>> a1 = pp.euler2SO3(torch.Tensor([0., 0., 0.]))
        >>> a2 = pp.euler2SO3(torch.Tensor([torch.pi / 4., torch.pi / 3., torch.pi / 2.]))
        >>> poses = pp.SE3([[[0., 4., 0., a1[0], a1[1], a1[2], a1[3]],
        ...                  [0., 3., 0., a1[0], a1[1], a1[2], a1[3]],
        ...                  [0., 2., 0., a1[0], a1[1], a1[2], a1[3]],
        ...                  [0., 1., 0., a1[0], a1[1], a1[2], a1[3]],
        ...                  [1., 0., 1., a2[0], a2[1], a2[2], a2[3]],
        ...                  [2., 0., 1., a2[0], a2[1], a2[2], a2[3]],
        ...                  [3., 0., 1., a2[0], a2[1], a2[2], a2[3]],
        ...                  [4., 0., 1., a2[0], a2[1], a2[2], a2[3]]]])
        >>> wayposes = pp.bspline(poses, 0.1)

    .. figure:: /_static/img/module/liespline/BsplineSE3.png

        Fig. 1. Result of B Spline Interpolation in SE3.
    '''
    assert is_SE3(data), "The input poses are not SE3Type."
    assert data.dim() >= 2, "Dimension of data should be [..., N, C]."
    assert interval < 1.0, "The interval should be smaller than 1."
    batch = data.shape[:-2]
    if extrapolate:
        data = torch.cat((data[..., :1, :].expand(batch+(2,-1)),
                          data,
                          data[..., -1:, :].expand(batch+(2,-1))), dim=-2)
    else:
        assert data.shape[-2]>=4, "Number of poses is less than 4."
    Bth, N, D = data.shape[:-2], data.shape[-2], data.shape[-1]
    dargs = {'dtype': data.dtype, 'device': data.device}
    timeline = torch.arange(0, 1, interval, **dargs)
    tt = timeline ** torch.arange(4, **dargs).view(-1, 1)
    B = torch.tensor([[5, 3,-3, 1],
                      [1, 3, 3,-2],
                      [0, 0, 0, 1]], **dargs) / 6
    dP = data[..., 0:-3, :].unsqueeze(-2)
    w = (B @ tt).unsqueeze(-1)
    index = (torch.arange(0, N-3).unsqueeze(-1) + torch.arange(0, 4)).view(-1)
    P = data[..., index, :].view(Bth + (N-3,4,1,D))
    P = (P[..., :3, :, :].Inv() * P[..., 1:, :, :]).Log()
    A = (P * w).Exp()
    Aend = (P[..., -1, :] * ((B.sum(dim=1)).unsqueeze(-1))).Exp()
    Aend = Aend[..., [0], :] * Aend[..., [1], :] * Aend[..., [2], :]
    A = A[..., 0, :, :] * A[..., 1, :, :] * A[..., 2, :, :]
    ps, pend = dP * A,dP[...,-1,:,:]*Aend[...,-1,:,:]
    poses = torch.cat((ps.view(Bth + (-1, D)), pend), dim=-2)
    return poses
