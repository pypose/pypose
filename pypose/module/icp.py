import torch
from ..module.pnp import EPnP

class ICP(torch.nn.Module):
    r'''
    Iterative Closest Point (ICP) using Singular Value Decomposition (SVD).
    Args:
        steplim: the max step number
        tol: the tolerance of error for early stopping
        matched: whether the input points set and the target points set have been matched
    '''

    def __init__(self, steplim=200, tol=0.0001, init_transform =None):
        super().__init__()
        assert type(init_transform) == pp.LieTensor
        self.steplim = steplim
        self.tol = tol
        self.init_transform = init_transform

    def _k_nearest_neighbor(self, pc1, pc2, k = 1, norm = 2, sort: bool = True):

        r'''
        Select the k nearest neighbor point of pc1 from pc2
        Args:
            pc1: the source points set
            pc2: the target points set
            tolerance: the threshold of min distance
        Returns:
            nn.values: the min distance between point in p1 and its nearest neighbor
            nn.indices: the index of the nearest neighbor point in p2
        '''

        diff = pc1.unsqueeze(-2) - pc2.unsqueeze(-3)
        if norm == 1:
            distance = torch.sum(torch.abs(diff), dim=-1)
        elif norm == 2:
            distance = torch.sqrt(torch.sum(diff ** 2, dim=-1))
        nn = distance.topk(k, largest=False)

        # Will add sorted function later.

        return nn

    def forward(self, p1, p2):
        temppc = p1.clone()
        iter = 0
        err = 0
        if (self.init_transform != None):
            temppc = self.init_transform.Act(temppc)
        while iter <= self.steplim:
            iter += 1
            nn = self._k_nearest_neighbor(temppc, p2)
            errnew = sum(sum(sum(nn.values) / len(nn.values)))
            T = EPnP._points_transform(temppc, p2[:, nn.indices[-1],:].squeeze(-2))
            temppc = T.unsqueeze(-2).Act(temppc)
            if (abs(err - errnew) < self.tol):
                break
            err = errnew
        T = EPnP._points_transform(p1, temppc)
