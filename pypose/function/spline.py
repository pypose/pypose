import torch
from . import is_SE3


def chspline(points, interval=0.1):
    r"""
    Cubic Hermite Spline, a piecewise-cubic interpolator matching values and first
    derivatives.

    Args:
        points (:obj:`Tensor`): the sequence of points for interpolation with shape
            [..., point_num, dim].
        interval (:obj:`Float`): the interval between adjacant interpolation points.

    Returns:
       ``Tensor``: the interpolated points with [..., inter_points_num, dim] shape.

    On the unit interval [0, 1], given a starting point :math:`p_0` at :math:`t = 0` and
    an ending point :math:`p_1` at :math:`t = 1` with starting tangent :math:`m_0` at
    :math:`t = 0` and ending tagent :math:`m_1` at :math:`t=1`, the polynomial can be
    defined by

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
        >>> waypoints = pp.chspline(points)

    .. figure:: /_static/img/module/liespline/CsplineR3.png
        :width: 600

        Fig. 1. Result of Cubic Spline Interpolation in 3D space.
    """
    assert points.dim() >= 2, 'Dimension of points should be [..., N, C]'
    assert 1/interval%1 == 0.0, '1 should be divisible by interval between points'
    if points.dim()==2:
        points = points.unsqueeze(0)
    batch, N = points.shape[:-2], points.shape[-2]
    dargs = {'device': points.device, 'dtype': points.dtype}
    xs = torch.arange(0, N-1+interval, interval, **dargs)
    x = torch.arange(N, **dargs)
    idxs = torch.searchsorted(x[...,1:], xs[..., :])
    x = x.expand(batch+(-1,))
    xs = xs.expand(batch+(-1,))
    m = (points[..., 1:, :] - points[..., :-1, :])
    m /= (x[..., 1:] - x[..., :-1])[..., None]
    m = torch.cat([m[...,[0],:], (m[..., 1:,:] + m[..., :-1,:]) / 2, m[...,[-1],:]], -2)
    idxs = torch.searchsorted(x[0, 1:], xs[0, :])
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


def bspline(data, timeline):
    r'''
    B-spline interpolation, which currently only support SE3 LieTensor.

    Args:
        data (:obj:`LieTensor`): the input sparse poses with
            [batch_size, point_num, dim] shape.
        timeline (:obj:`Tensor`): the time line to interpolate.

    Returns:
        ``LieTensor``: the interpolated SE3 LieTensor.

    A poses query at any time :math:`t \in [t_i, t_{i+1})` (i.e. a segment of the spline)
    only relies on the poses located at time steps :math:`\{t_{i-1},t_i,t_{i+1},t_{i+2}\}`.
    It means that the interpolation between adjacent poses needs four consecutive poses.
    Thus, the B-spline interpolation could estimate the pose between :math:`[t_1, t_{n-1}]`
    by the input poses at :math:`\{t_0, ...,t_{n}\}`.

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
        >>> timeline = torch.arange(0, 1, 0.25)
        >>> poses = pp.SE3([[[0., 4., 0., a1[0], a1[1], a1[2], a1[3]],
        ...                  [0., 3., 0., a1[0], a1[1], a1[2], a1[3]],
        ...                  [0., 2., 0., a1[0], a1[1], a1[2], a1[3]],
        ...                  [0., 1., 0., a1[0], a1[1], a1[2], a1[3]],
        ...                  [1., 0., 1., a2[0], a2[1], a2[2], a2[3]],
        ...                  [2., 0., 1., a2[0], a2[1], a2[2], a2[3]],
        ...                  [3., 0., 1., a2[0], a2[1], a2[2], a2[3]],
        ...                  [4., 0., 1., a2[0], a2[1], a2[2], a2[3]]]])
        >>> wayposes = pp.bspline(poses, timeline)

    - .. figure:: /_static/img/module/liespline/BsplineSE3.png

        Fig. 1. Result of B Spline Interpolation in SE3.
    '''
    assert is_SE3(data), "The input poses are not SE3Type."
    data = torch.cat((data[..., :1, :], data, data[..., -1:, :]), dim=-2)
    Bth, N, D, K = data.shape[:-2], data.shape[-2], data.shape[-1], timeline.shape[-1]
    dargs = {'dtype': timeline.dtype, 'device': timeline.device}
    tt = timeline ** torch.arange(4, **dargs).view(-1, 1)
    B = torch.tensor([[5, 3,-3, 1],
                      [1, 3, 3,-2],
                      [0, 0, 0, 1]], **dargs) / 6
    dP = data[..., 0:-3, :].unsqueeze(-2)
    w = (B @ tt).unsqueeze(-1)
    index = (torch.arange(0, N-3).unsqueeze(-1) + torch.arange(0, 4)).view(-1)
    P = data[..., index, :].view(Bth + (N-3, 4, 1, D))
    A = ((P[..., :3,:, :].Inv() * P[..., 1:,:,:]).Log() * w).Exp()
    A = A[...,0,:,:] * A[...,1,:,:] * A[...,2,:,:]
    return (dP * A).view(Bth + (K * (N - 3), D))
