import torch
from .. import knn, points_transform

class ICP(torch.nn.Module):
    r'''
    Iterative Closest Point (ICP) using Singular Value Decomposition (SVD).
    
    Args:
        steplim: the max step number
        tol: the tolerance of error for early stopping
        tf: initial transformation

    ICP algorithm aims to calculate the optimal transformation :math:`T`
    between two point clouds, source point cloud and target point cloud.

    .. math::
        \begin{align*}
            argmin \sum \|p_i^t - Tp_i^s\|,
        \end{align*}

    where :math:`p^t` is the point in target point cloud, and :math:`p^s` is
    the point in the source point cloud.

    1. For each point in pc1, nearest neighbor algorithm is used to select its
    closest point in pc2 to the matched point pairs.

    2. Singular value decomposition (SVD) algorithm is used to get the rotation
    and translation matrices from the matched point pairs.

    3. The source point cloud is updated using the obtained rotation and
    translation matrices. The distance between the updated source point cloud and
    the target point cloud is calculated.

    4. Iterate these steps until the change of obtained distance is lower than the
    given tolerance.
    
    '''

    def __init__(self, steplim = 200, tol = 0.0001, tf = None):
        super().__init__()
        self.steplim = steplim
        self.tol = tol
        self.tf = tf

    def forward(self, p1, p2):
        r'''

        Args:
            p1(``torch.Tensor``): the source point cloud with [..., points_num, 3] shape.
            p2(``torch.Tensor``): the target point cloud with [..., points_num, 3] shape.

        Returns:
            ``LieTensor``: the estimated transformation from p1 to p2.

        '''
        temppc = p1.clone()
        iter = 0
        err = 0
        if self.tf != None:
            if p1.shape[:-2] == self.tf.shape[:-1] and \
                self.tf.shape[-1] == 7:
                temppc = self.tf.unsqueeze(-2).Act(temppc)
            else:
                raise ValueError("Invalid initial transformation matrix, please use " +
                                 "SE3 LieTensor with the same batch sizes as the " +
                                 "input pointcloud.")
        while iter <= self.steplim:
            iter += 1
            neighbors = knn(temppc, p2)
            knndist = neighbors.values.squeeze(-1)
            knnidx = neighbors.indices
            errnew = torch.mean(knndist, dim=-1)
            if torch.all(torch.abs(errnew - err) < self.tol):
                break
            err = errnew
            batch = torch.arange(temppc.shape[0],dtype=knnidx.dtype, device=knnidx.device).repeat(knnidx.shape[1],1).T.unsqueeze(-1)
            batch_knnidx = torch.cat((batch, knnidx),dim =-1)
            T = points_transform(temppc, p2[batch_knnidx[:,:,0],batch_knnidx[:,:,1],:])
            temppc = T.unsqueeze(-2).Act(temppc)
        T = points_transform(p1, temppc)
        return T
