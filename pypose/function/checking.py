import torch, math
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


def hasnan(obj:list):
    r'''
    Checks whether a deep nested list of tensors contains Nan or not.

    Args:
        obj (``obj``): a Python object that can be a list of nested list.

    Return:
        ``bool``: ``True`` if the list contains a tensor with ``Nan`` otherwise ``False``.

    Example:
        >>> L1 = [[1, 3], [4, [5, 6]], 7, [8, torch.tensor([0, -1.0999])]]
        >>> hasnan(L1)
        False
        >>> L2 = [[torch.tensor([float('nan'), -1.0999]), 3], [4, [5, 6]], 7, [8, 9]]
        >>> hasnan(L2)
        True
        >>> L3 = [[torch.tensor([1, -1.0999]), 3], [4, [float('nan'), 6]], 7, [8, 9]]
        >>> hasnan(L3)
        True
    '''
    if isinstance(obj, list) or isinstance(obj, tuple):
        for l in obj:
            if hasnan(l):
                return True
        return False
    else:
        return torch.isnan(obj).any() if torch.is_tensor(obj) else math.isnan(obj)
