import torch
from . import EPnP
from .. import homo2cart, cart2homo


class ICP(torch.nn.Module):
    r'''
    Iterative Closest Point (ICP) using Singular Value Decomposition (SVD).

    Args:
        steplim: the max step number
        tol: the tolerance of error for early stopping
        matched: whether the input points set and the target points set have been matched
    '''

    def __init__(self, steplim=200, tol=0.0001, matched=False):
        super().__init__()
        self.steplim = steplim
        self.tol = tol
        self.matched = matched

    def forward(self, newpc, originpc):
        temppc = newpc.clone()
        iter = 0
        err = 0
        if (not self.matched):
            while iter <= self.steplim:
                iter += 1
                neighbor = self.nearest_neighbor(temppc, originpc)
                errnew = neighbor.values.mean()
                tf = EPnP._points_transform(temppc, originpc).matrix().unsqueeze(-3)
                temppc = homo2cart((tf @ cart2homo(temppc).unsqueeze(-1)).squeeze(-1))
                if (abs(err - errnew) < self.tol):
                    break
                err = errnew
            T = EPnP._points_transform(newpc, temppc)
        else:
            T = EPnP._points_transform(newpc, originpc)
        return T

    def nearest_neighbor(self, p1, p2, k=1):
        r'''
        Select the nearest neighbor point of p1 from p2
        Args:
            p1: the source points set
            p2: the target points set
            tolerance: the threshold of min distance

        Returns:
            distances: the min distance between point in p1 and its nearest neighbor
            indices: the index of the nearest neighbor point in p2
        '''
        dif = torch.stack([p2[i].unsqueeze(-2) - p1[i]
                           for i in range(p1.shape[0])])

        dist = torch.norm(dif, dim=-1)
        nn = dist.topk(k, largest=False)
        return nn
