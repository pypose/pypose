import torch
import numpy as np


def print_header(msg):
    print('===>', msg)


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().numpy()


def bger(x, y):
    return x.unsqueeze(2).bmm(y.unsqueeze(1))


def get_sizes(G, A=None):
    if G.dim() == 2:
        nineq, nz = G.size()
        nBatch = 1
    elif G.dim() == 3:
        nBatch, nineq, nz = G.size()
    if A is not None:
        neq = A.size(1) if A.nelement() > 0 else 0
    else:
        neq = None
    # nBatch = batchedTensor.size(0) if batchedTensor is not None else None
    return nineq, nz, neq, nBatch


def bdiag(d):
    nBatch, sz = d.size()
    D = torch.zeros(nBatch, sz, sz).type_as(d)
    I = torch.eye(sz).repeat(nBatch, 1, 1).type_as(d).bool()
    D[I] = d.squeeze().view(-1)
    return D


def expandParam(X, nBatch, nDim):
    if X.ndimension() in (0, nDim) or X.nelement() == 0:
        return X, False
    elif X.ndimension() == nDim - 1:
        return X.unsqueeze(0).expand(*([nBatch] + list(X.size()))), True
    else:
        raise RuntimeError("Unexpected number of dimensions.")


def extract_nBatch(Q, p, G, h, A, b):
    dims = [3, 2, 3, 2, 3, 2]
    params = [Q, p, G, h, A, b]
    for param, dim in zip(params, dims):
        if param.ndimension() == dim:
            return param.size(0)
    return 1
