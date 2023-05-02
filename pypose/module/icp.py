import torch
from .. import knn, svdtf, is_SE3

class ICP(torch.nn.Module):
    r'''
    Batched Iterative Closest Point (ICP) algorithm to find a rigid transformation
    between two sets of points using Singular Value Decomposition (SVD).

    Args:
        steps (``int``, optional): the maximum number of ICP iteration steps. Default: 200.
        tol (``double``, optional): the tolerance of the relative error used to terminate
            the algorithm. Default: 1e-6.
        init (``LieTensor``, optional): the initial transformation :math:`T_{init}` in
            ``SE3type``. Default: ``None``.

    The algorithm takes two input point clouds: source point cloud and target point cloud.
    The objective is to find the optimal similarity transformation ( :math:`T` ) to
    minimize the error between the transformed source point cloud and the target
    point cloud as shown in the equation:

    .. math::
        \begin{align*}
            \underset{T}{\operatorname{arg\,min}} \sum_i \| p_{\mathrm{target, j}} -
            T \cdot p_{\mathrm{source, i}}\|,
        \end{align*}

    where :math:`p_{\mathrm{source, i}}` is the ith point in the source point cloud, and
    :math:`p_{\mathrm{target, j}}` is the cloest point to :math:`p_{\mathrm{source, i}}`
    in the target point cloud with index j. The algorithm consists of the following steps:

    1. For each point in source, the nearest neighbor algorithm (KNN) is used to select
    its closest point in target to form the matched point pairs.

    2. Singular value decomposition (SVD) algorithm is used to compute the rotation
    and translation matrices from the matched point pairs.

    3. The source point cloud is updated using the obtained rotation and translation
    matrices. The distance between the updated source and target is calculated.

    4. The algorithm continues to iterate through these steps until the change in the
    calculated distance falls below the specified tolerance level or the maximum number
    of iteration steps is reached.

    Example:
        >>> import torch, pypose as pp
        >>> source = torch.tensor([[[0., 0., 0.],
        ...                         [1., 0., 0.],
        ...                         [2.,  0, 0.]]])
        >>> target = torch.tensor([[[0.2,      0.1,  0.],
        ...                         [1.1397, 0.442,  0.],
        ...                         [2.0794, 0.7840, 0.]]])
        >>> icp = pp.module.ICP()
        >>> icp(source, target)
        SE3Type LieTensor:
        LieTensor([[0.2000, 0.1000, 0.0000, 0.0000, 0.0000, 0.1736, 0.9848]])

    Warning:
        It's important to note that the solution is sensitive to the initialization.
    '''
    def __init__(self, steps=200, tol=1e-6, init=None):
        super().__init__()
        assert init is None or is_SE3(init), "The initial transformation is not SE3Type."
        self.steps, self.tol, self.init = steps, tol, init

    def forward(self, source, target, ord=2, dim=-1, init=None):
        r'''
        Args:
            source (``torch.Tensor``): the source point clouds with shape
                (..., points_num, 3).
            target (``torch.Tensor``): the target point clouds with shape
                (..., points_num, 3).
            ord (``int``, optional): the order of norm to use for distance calculation.
                Default: ``2`` (Euclidean distance).
            dim (``int``, optional): the dimension encompassing the point cloud
                coordinates, utilized for calculating distance.
                Default: ``-1`` (The last dimension).
            init (``LieTensor``, optional): the initial transformation :math:`T_{init}` in
                ``SE3type``. If not ``None``, it will suppress the ``init`` given by the
                class constructor. Default: ``None``.

        Returns:
            ``LieTensor``: The estimated transformation (``SE3type``) from source to
            target point cloud.
        '''
        temporal, errlast = source, 0
        init = init if init is not None else self.init
        batch = torch.broadcast_shapes(source.shape[:-2], target.shape[:-2])
        if init is not None:
            assert is_SE3(init), "The initial transformation is not SE3Type."
            temporal = init.unsqueeze(-2) @ temporal
        for _ in range(self.steps):
            knndist, knnidx = knn(temporal, target, k=1, ord=ord, dim=dim)
            errnew = knndist.squeeze(-1).mean(dim=-1)
            if torch.all((errnew - errlast).abs() < self.tol):
                break
            errlast = errnew
            target = target.expand(batch + target.shape[-2:])
            knntarget = torch.gather(target, -2, knnidx.expand(batch + source.shape[-2:]))
            T = svdtf(temporal, knntarget)
            temporal = T.unsqueeze(-2) @ temporal
        return svdtf(source, temporal)
