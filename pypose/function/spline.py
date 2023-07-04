import torch
from . import is_SE3


def chspline(points, num=10):
    r"""
    Cubic Hermite Spline, a piecewise-cubic interpolator matching values and first
    derivatives.

    Args:
        points (:obj:`Tensor`): the sequence of points for interpolation with shape
            [..., point_num, dim].
        num (:obj:`int`): the number of interpolated points between adjcent two poses.
            Default: ``10``.  

    Returns:
       ``Tensor``: the interpolated points.

    The interpolated points are evenly distributed between a start point and a end point
    according to the number of interpolated points. In this function, the interpolation
    time is aslo evenly distributed between the corresponding time of start point and
    end point. For example, if 10 points are interpolated between the points at
    :math:`t_0` and :math:`t_0+1`. The interpolation time is
    :math:`t\in[t_0, t_0+0.1,...,t_0+0.9]`.

    Denote the starting point :math:`p_0` with the starting tangent :math:`m_0` and the
    ending point :math:`p_1` with the ending tagent :math:`m_1`. The interpolated point at
    time :math:`t` can be defined as:

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
        >>> waypoints = pp.chspline(points, 10)

    .. figure:: /_static/img/module/liespline/CsplineR3.png
        :width: 600

        Fig. 1. Result of Cubic Spline Interpolation in 3D space.
    """
    assert points.dim() >= 2, "Dimension of points should be [..., N, C]"
    assert type(num)==int, "The type of interpolated point number should be int."
    assert num > 0, "The number of interpolated point number should be larger than 0."
    batch, N = points.shape[:-2], points.shape[-2]
    dargs = {'device': points.device, 'dtype': points.dtype}
    interval = 1.0/num
    timeline = interval*torch.arange(0, num*(N-1)+1, **dargs)
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


def bspline(data, num:int=10, clamping=False):
    r'''
    B-spline interpolation, which currently only supports SE3 LieTensor.

    Args:
        data (:obj:`LieTensor`): the input sparse poses with
            [batch_size, poses_num, dim] shape.
        num (``int``): the number of interpolated poses between adjcent two poses.
            Default: ``10``.
        clamping(``bool``): flag to determine whether the interpolate poses pass the 
            start and end poses. If ``True`` the interpolated poses pass the start and
            end poses. Default: ``False``.

    Returns:
        :obj:`LieTensor`: the interpolated SE3 LieTensor.

    A poses query at any time :math:`t \in [t_i, t_{i+1})` (i.e. a segment of the spline)
    only relies on the poses at four time steps :math:`\{t_{i-1},t_i,t_{i+1},t_{i+2}\}`.
    It means that the interpolation between adjacent poses needs four consecutive poses.
    In this function, the interpolation time :math:`t` is evenly distributed between a
    poses query according to the number of interpolated poses.

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
        >>> wayposes = pp.bspline(poses, 10)

    - .. figure:: /_static/img/module/liespline/BsplineSE3.png

        Fig. 1. Result of B Spline Interpolation in SE3.
    '''
    assert is_SE3(data), "The input poses are not SE3Type."
    assert data.dim() >= 2, "Dimension of data should be [..., N, C]."
    assert type(num) == int, "The type of interpolated pose number should be int."
    assert num > 0, "The number of interpolated pose number should be larger than 0."
    batch = data.shape[:-2]
    if clamping:
        data = torch.cat((data[..., :1, :].expand(batch+(3,-1)),
                          data,
                          data[..., -1:, :].expand(batch+(3,-1))), dim=-2)
    else:
        data = torch.cat((data[..., :1, :],
                          data,
                          data[..., -1:, :]), dim=-2)
    Bth, N, D = data.shape[:-2], data.shape[-2], data.shape[-1]
    dargs = {'dtype': data.dtype, 'device': data.device}
    timeline = torch.arange(0, 1, 1.0/num, **dargs)
    tt = timeline ** torch.arange(4, **dargs).view(-1, 1)
    B = torch.tensor([[5, 3,-3, 1],
                      [1, 3, 3,-2],
                      [0, 0, 0, 1]], **dargs) / 6
    dP = data[..., 0:-3, :].unsqueeze(-2)
    w = (B @ tt).unsqueeze(-1)
    index = (torch.arange(0, N-3).unsqueeze(-1) + torch.arange(0, 4)).view(-1)
    P = data[..., index, :].view(Bth + (N-3,4,1,D))
    A = ((P[..., :3,:, :].Inv() * P[...,1:,:,:]).Log() * w).Exp()
    A = A[...,0,:,:] * A[...,1,:,:] * A[...,2,:,:]
    return (dP * A).view(Bth + (num * (N - 3), D))