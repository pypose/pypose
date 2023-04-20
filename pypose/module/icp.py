import torch
from . import EPnP
from .. import knn

class ICP(torch.nn.Module):
    r'''
    Iterative Closest Point (ICP) using Singular Value Decomposition (SVD).
    Args:
        steplim: the max step number
        tol: the tolerance of error for early stopping
        tf: initial transformation
    '''

    def __init__(self, steplim = 200, tol = 0.0001, tf = None):
        super().__init__()
        self.steplim = steplim
        self.tol = tol
        self.tf = tf

    def forward(self, p1, p2):
        temppc = p1.clone()
        iter = 0
        err = None
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
            knndist, knnidx = knn(temppc, p2)
            knndist = knndist.squeeze(-1)
            knnidx = knnidx.squeeze(-1)
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
