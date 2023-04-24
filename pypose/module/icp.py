import torch
from .. import knn, points_transform

class ICP(torch.nn.Module):
    r'''
    This class implements the batched Iterative Closest Point (ICP) algorithm to find a
    similarity transformation (:math:`T`) between two differently-sized sets of
    3-dimensional points using Singular Value Decomposition (SVD). It's important to note
    that the solution found is only a local optimum.

    Args:
        steplim (``int``, optional): The maximum number of ICP iteration steps.
            Default: 200.
        tol (``double``, optional): The tolerance of the relative error used to terminate
            the algorithm. Default: 1e-4.
        tf (``LieTensor``, optional): The initial transformation :math:`T_{init}` in
            ``SE3type``. Default: None.

    The algorithm takes two input point clouds: source point cloud (psrc) and target
    point cloud (ptgt). The objective is to find the optimal similarity transformation
    (:math:`T`) to minimize the error between the transformed source point cloud and the
    target point cloud as shown in the equation:

    .. math::
        \begin{align*}
            \operatorname{arg\,min}_T \sum_i \|ptgt_i - T \cdot psrc_j\|,
        \end{align*}

    where :math:`ptgt_i` is the ith point in the target point cloud, and :math:`psrc_j` is
    the jth point in the source point cloud. The algorithm consists of the following steps:

    1. For each point in psrc, the nearest neighbor algorithm is used to select its
    closest point in ptgt to form the matched point pairs.

    2. Singular value decomposition (SVD) algorithm is used to compute the rotation
    and translation matrices from the matched point pairs.

    3. The source point cloud is updated using the obtained rotation and
    translation matrices. The distance between the updated source point cloud and
    the target point cloud is calculated.

    4. The algorithm iterates through these steps until the change in the obtained
    distance is lower than the given tolerance.

    '''

    def __init__(self, steplim = 200, tol = 1e-4, tf = None):
        super().__init__()
        self.steplim = steplim
        self.tol = tol
        self.tf = tf

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


        temppc = psrc.clone()
        iter = 0
        err = 0
        if self.tf != None:
            if psrc.shape[:-2] == self.tf.shape[:-1] and \
                self.tf.shape[-1] == 7:
                temppc = self.tf.unsqueeze(-2).Act(temppc)
            else:
                raise ValueError("Invalid initial transformation matrix, please use " +
                                 "SE3 LieTensor with the same batch sizes as the " +
                                 "input pointcloud.")
        while iter <= self.steplim:
            iter += 1
            neighbors = knn(temppc, ptgt)
            knndist = neighbors.values.squeeze(-1)
            knnidx = neighbors.indices
            errnew = torch.mean(knndist, dim=-1)
            if torch.all(torch.abs(errnew - err) < self.tol):
                break
            err = errnew
            batch = torch.arange(temppc.shape[0],dtype=knnidx.dtype, device=knnidx.device).repeat(knnidx.shape[1],1).T.unsqueeze(-1)
            batch_knnidx = torch.cat((batch, knnidx),dim =-1)
            T = points_transform(temppc, ptgt[batch_knnidx[:,:,0],batch_knnidx[:,:,1],:])
            temppc = T.unsqueeze(-2).Act(temppc)
        T = points_transform(psrc, temppc)
        return T
