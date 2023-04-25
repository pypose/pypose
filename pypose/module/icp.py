import torch
from .. import lietensor
from .. import knn, svdtf

class ICP(torch.nn.Module):
    r'''
    Batched Iterative Closest Point (ICP) algorithm to find a rigid transformation
    between two sets of points using Singular Value Decomposition (SVD).

    Args:
        steps (``int``, optional): The maximum number of ICP iteration steps. Default: 200.
        tol (``double``, optional): The tolerance of the relative error used to terminate
            the algorithm. Default: 1e-6.
        tf (``LieTensor``, optional): The initial transformation :math:`T_{init}` in
            ``SE3type``. Default: None.

    The algorithm takes two input point clouds: source point cloud (psrc) and target
    point cloud (ptgt). The objective is to find the optimal similarity transformation
    ( :math:`T` ) to minimize the error between the transformed source point cloud and the
    target point cloud as shown in the equation:

    .. math::
        \begin{align*}
            \underset{T}{\operatorname{arg\,min}} \sum_i \| p_{\mathrm{target, j}} -
            T \cdot p_{\mathrm{source, i}}\|,
        \end{align*}

    where :math:`p_{\mathrm{source, i}}` is the ith point in the source point cloud, and
    :math:`p_{\mathrm{target, j}}` is the cloest point to :math:`p_{\mathrm{source, i}}`
    in the target point cloud with index j. The algorithm consists of the following steps:

    1. For each point in psrc, the nearest neighbor algorithm (knn) is used to select its
    closest point in ptgt to form the matched point pairs.

    2. Singular value decomposition (SVD) algorithm is used to compute the rotation
    and translation matrices from the matched point pairs.

    3. The source point cloud (psrc) is updated using the obtained rotation and
    translation matrices. The distance between the updated psrc and ptgt is calculated.

    4. The algorithm continues to iterate through these steps until the change in the
    calculated distance falls below the specified tolerance level or the maximum number
    of iteration steps is reached.

    Example:
        >>> import torch, pypose as pp
        >>> pc1 = torch.tensor([[[0., 0., 0.],
        ...                      [1., 0., 0.],
        ...                      [2., 0, 0.]]])
        >>> pc2 = torch.tensor([[[0.2, 0.1, 0.],
        ...                      [1.1397, 0.442, 0.],
        ...                      [2.0794, 0.7840, 0.]]])
        >>> icp = pp.module.ICP()
        >>> icp(pc1, pc2)
        SE3Type LieTensor:
        LieTensor([[0.2000, 0.1000, 0.0000, 0.0000, 0.0000, 0.1736, 0.9848]])

    Warning:
        It's important to note that the solution found is only a local optimum.

    '''
    def __init__(self, steps=200, tol=1e-6, init=None):
        super().__init__()
        self.steps = steps
        self.tol = tol
        self.init = init
        if init != None:
            assert isinstance(init.ltype, lietensor.lietensor.SE3Type), "The input \
                initial transformation is not of type SE3Type."

    def forward(self, psrc, ptgt):
        r'''
        Args:
            psrc(``torch.Tensor``): The source point cloud tensor with
                [..., points_num, 3] shape.
            ptgt(``torch.Tensor``): The target point cloud tensor with
                [..., points_num, 3] shape.

        Returns:
            ``LieTensor``: The estimated transformation (``SE3type``) from psrc to ptgt.

        '''
        temppc = psrc
        err = 0
        dim = psrc.shape
        for i in range(self.steps):
            neighbors = knn(temppc, ptgt)
            knndist = neighbors.values.squeeze(-1)
            knnidx = neighbors.indices
            errnew = knndist.mean(dim=-1)
            if torch.all(torch.abs(errnew - err) < self.tol):
                break
            err = errnew
            ptgtknn = torch.gather(ptgt, -2, knnidx.expand(dim))
            T = svdtf(temppc, ptgtknn)
            temppc = T.unsqueeze(-2).Act(temppc)
        T = svdtf(psrc, temppc)
        return T
