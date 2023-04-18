import torch
from . import EPnP
from .. import LieTensor

class ICP(torch.nn.Module):
    r'''
    Iterative Closest Point (ICP) using Singular Value Decomposition (SVD).
    Args:
        steplim: the max step number
        tol: the tolerance of error for early stopping
        matched: whether the input points set and the target points set have been matched
    '''

    def __init__(self, steplim=200, tol=1e-6, init_transform =None):
        super().__init__()
        #assert type(init_transform) == LieTensor
        self.steplim = steplim
        self.tol = tol
        self.init_transform = init_transform

    def forward(self, p1, p2):
        temppc = p1.clone()
        iter = 0
        err = None
        if (self.init_transform != None):
            temppc = self.init_transform.Act(temppc)
        while iter <= self.steplim:
            iter += 1
            neighbors = self._k_nearest_neighbor(temppc, p2)
            errnew = torch.mean(neighbors.values, dim=-1)
            T = EPnP._points_transform(temppc, p2[:, neighbors.indices[-1],:].squeeze(-2))
            temppc = T.unsqueeze(-2).Act(temppc)
            if err is None:
                err = errnew
            else:
                if torch.all(torch.abs((errnew - err) / err) < self.tol):
                    break
            err = errnew
        T = EPnP._points_transform(p1, temppc)
        return T

    @staticmethod
    def _k_nearest_neighbor(pc1, pc2, k = 1, norm = 2, sort: bool = False):
        r'''
        Select the k nearest neighbor point of pc1 from pc2
        Args:
            pc1: the source points set
            pc2: the target points set
            k: the number of nearest neighbors to find
            norm: the norm to use for distance calculation (1 or 2)
            sort: whether to sort the k nearest neighbors by distance

        Returns:
            distances: the distance between each point in pc1 and
                        its k nearest neighbors in pc2
            indices: the index of the k nearest neighbor points in pc2
        '''

        diff = pc1.unsqueeze(-2) - pc2.unsqueeze(-3)
        distance = torch.linalg.norm(diff, dim=-1, ord=norm)
        knn = distance.topk(k, largest=False)

        if k > 1 and sort:
            dist, rank= knn.values.sort(dim=-1)
            idx = torch.gather(knn.indices, -1, rank)
            TopK = torch.namedtuple('TopK', ['values', 'indices'])
            knn = TopK(values=dist, indices=idx)

        return knn
