import torch
from .. import lietensor
from .. import LieTensor


def is_lietensor(obj):
    r'''
    Check whether an instance or object is a LieTensor or not.

    Args:
        obj (``obj``): a Python object or instantance.

    Return:
        ``bool``: ``True`` if obj is a LieTensor object otherwise ``False``.
    '''
    return True if isinstance(obj, LieTensor) else False


def is_SE3(obj):
    r'''
    Check whether an instance or object is an SE3 Type LieTensor or not.

    Args:
        obj (``obj``): a Python object or instantance.

    Return:
        ``bool``: ``True`` if obj is a SE3 Type LieTensor object otherwise ``False``.
    '''
    return True if isinstance(obj.ltype, lietensor.lietensor.SE3Type) else False


def hasnan(L:list):
    r'''
    Checks whether a deep nested list of tensors contains Nan or not.
    '''
    if isinstance(L, list) or isinstance(L, tuple):
        for l in L:
            if hasnan(l):
                return True
        return False
    else:
        return torch.isnan(L).any() if torch.is_tensor(L) else False
