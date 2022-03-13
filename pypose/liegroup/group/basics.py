import torch


def vec2skew(v):
    """Batch Skew Matrix"""
    assert v.shape[-1] == 3, "Last dim should be 3"
    shape, v = v.shape, v.view(-1,3)
    S = torch.zeros(v.shape[:-1]+(3,3), device=v.device, dtype=v.dtype)
    S[:,0,1], S[:,0,2] = -v[:,2],  v[:,1]
    S[:,1,0], S[:,1,2] =  v[:,2], -v[:,0]
    S[:,2,0], S[:,2,1] = -v[:,1],  v[:,0]
    return S.view(shape[:-1]+(3,3))
