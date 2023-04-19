import torch
from . import EPnP

class ICP(torch.nn.Module):
    r'''
    Iterative Closest Point (ICP) using Singular Value Decomposition (SVD).
    Args:
        steplim: the max step number
        tol: the tolerance of error for early stopping
        matched: whether the input points set and the target points set have been matched
    '''

    def __init__(self, steplim=200, tol=0.0001, init_transform = None):
        super().__init__()
        self.steplim = steplim
        self.tol = tol
        self.init_transform = init_transform

    def forward(self, p1, p2):
        temppc = p1.clone()
        iter = 0
        err = None
        if self.init_transform != None:
            if p1.shape[:-2] == self.init_transform.shape[:-1] and \
                self.init_transform.shape[-1] == 7:
                temppc = self.init_transform.unsqueeze(-2).Act(temppc)
            else:
                raise ValueError("Invalid initial transformation matrix, please use " +
                                 "SE3 LieTensor with the same batch sizes as the " +
                                 "input pointcloud.")
        while iter <= self.steplim:
            iter += 1
            knndist, knnidx = self._k_nearest_neighbor(temppc, p2)
            errnew = torch.mean(knndist, dim=-1)
            if err is None:
                err = errnew
            else:
                if torch.all(torch.abs((errnew - err) / err) < self.tol):
                    break
            err = errnew
            T = EPnP._points_transform(temppc, p2[:, knnidx[-1],:].squeeze(-2))
            temppc = T.unsqueeze(-2).Act(temppc)
        T = EPnP._points_transform(p1, temppc)
        return T

    @staticmethod
    def _k_nearest_neighbor(pc1, pc2, k = 1, norm = 2, sort: bool = False):
        r'''
        Select the k nearest neighbors point of pointcloud 1 from pointcloud 2 in each batch.

        Args:
            pc1 (``torch.Tensor``): The coordinates of the pointcloud 1.
                The shape has to be (..., N1, dim).
            pc2 (``torch.Tensor``): The coordinates of the pointcloud 2.
                The shape has to be (..., N2, dim).
            k (``int``, optional): The number of the nearest neighbors to be selected.
                k has to be k \seq N2. Default: ``1``.
            norm (``int``, optional): The norm to use for distance calculation.
                Default: ``2``.
            sort (``bool``, optional): Whether to sort the k nearest neighbors by distance.
                Default: ``False``.

        Returns:
            distance (``torch.Tensor``): The N-norm distance between each point in pc1 and
                its k nearest neighbors in pc2.
                The shape is (..., N1, k).
            indices (``torch.Tensor``): The index of the k nearest neighbor points in pc2
                The shape is (..., N1, k).
        '''

        diff = pc1.unsqueeze(-2) - pc2.unsqueeze(-3)
        dist = torch.linalg.norm(diff, dim=-1, ord=norm)
        knn = dist.topk(k, largest=False)

        if k > 1 and sort:
            distance, rank= knn.values.sort(dim=-1)
            indices = torch.gather(knn.indices, -1, rank)
        else:
            distance = knn.values
            indices = knn.indices
        return distance, indices
