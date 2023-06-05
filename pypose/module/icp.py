import torch
from .. import knn, svdtf, is_SE3
from ..utils.stepper import ReduceToBason


class ICP(torch.nn.Module):
    r'''
    Batched Iterative Closest Point (ICP) algorithm to find a rigid transformation
    between two sets of points using Singular Value Decomposition (SVD).

    Args:
        init (``LieTensor``, optional): the initial transformation :math:`T_{\text{init}}`
            in ``SE3type LieTensor``. Default: ``None``.
        stepper (``Planner``, optional): the stepper to stop a loop. If ``None``,
            the ``pypose.utils.ReduceToBason`` with a maximum of 200 steps are used.
            Default: ``None``.

    The algorithm takes two input point clouds (source and target) and finds the optimal
    transformation ( :math:`T` ) to minimize the error between the transformed source
    point cloud and the target point cloud:

    .. math::
        \begin{align*}
            \underset{T}{\operatorname{arg\,min}} \sum_i \| p_{\mathrm{target, j}} -
            T \cdot p_{\mathrm{source, i}}\|,
        \end{align*}

    where :math:`p_{\mathrm{source, i}}` is the i-th point in the source point cloud, and
    :math:`p_{\mathrm{target, j}}` is the cloest point of :math:`p_{\mathrm{source, i}}`
    in the target point clouds with index j. The algorithm consists of the following steps:

    1. For each point in source, the nearest neighbor algorithm (KNN) is used to select
       its closest point in target to form the matched point pairs.

    2. Singular value decomposition (SVD) algorithm is used to compute the transformation
       from the matched point pairs.

    3. The source point cloud is updated using the obtained transformation.
       The distance between the updated source and target is calculated.

    4. The algorithm continues to iterate through these steps until the ``stepper``
       condition is satisfied.

    Example:
        >>> import torch, pypose as pp
        >>> source = torch.tensor([[[0., 0., 0.],
        ...                         [1., 0., 0.],
        ...                         [2.,  0, 0.]]])
        >>> target = torch.tensor([[[0.2,      0.1,  0.],
        ...                         [1.1397, 0.442,  0.],
        ...                         [2.0794, 0.7840, 0.]]])
        >>> stepper = pp.utils.ReduceToBason(steps=10, verbose=True)
        >>> icp = pp.module.ICP(stepper=stepper)
        >>> icp(source, target)
        ReduceToBason step 0 loss tensor([0.4917])
        ReduceToBason step 1 loss tensor([7.4711e-08])
        ReduceToBason step 2 loss tensor([1.0450e-07])
        ReduceToBason step 3 loss tensor([2.8322e-07])
        ReduceToBason: Maximum patience steps reached, Quiting..
        SE3Type LieTensor:
        LieTensor([[0.2000, 0.1000, 0.0000, 0.0000, 0.0000, 0.1736, 0.9848]])

    Warning:
        It's important to note that the solution is sensitive to the initialization.
    '''
    def __init__(self, init=None, stepper=None):
        super().__init__()
        self.stepper = ReduceToBason(steps=200) if stepper is None else stepper
        assert init is None or is_SE3(init), "The initial transformation is not SE3Type."
        self.init = init

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
        temporal = source
        init = init if init is not None else self.init
        if init is not None:
            assert is_SE3(init), "The initial transformation is not SE3Type LieTensor."
            temporal = init.unsqueeze(-2) @ temporal
        batch = torch.broadcast_shapes(source.shape[:-2], target.shape[:-2])
        self.stepper.reset()
        while self.stepper.continual():
            knndist, knnidx = knn(temporal, target, k=1, ord=ord, dim=dim)
            error = knndist.squeeze(-1).mean(dim=-1)
            target = target.expand(batch + target.shape[-2:])
            knnidx = knnidx.expand(batch + source.shape[-2:])
            knntarget = torch.gather(target, -2, knnidx)
            T = svdtf(temporal, knntarget)
            temporal = T.unsqueeze(-2) @ temporal
            self.stepper.step(error)

        return svdtf(source, temporal)
