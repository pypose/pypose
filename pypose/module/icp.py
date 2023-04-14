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
                dist = torch.norm((originpc.unsqueeze(-2) - temppc.unsqueeze(-3)),dim=-1)
                errnew = dist.topk(1, largest=False).values.mean()
                tf = EPnP._points_transform(temppc, originpc).unsqueeze(-2)
                temppc = tf.Act(temppc)
                if (abs(err - errnew) < self.tol):
                    break
                err = errnew
            T = EPnP._points_transform(newpc, temppc)
        else:
            T = EPnP._points_transform(newpc, originpc)
        return T
