import math
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


def cumops_(v, dim, ops):
    L = v.shape[dim]
    assert dim != -1 or dim != v.shape[-1], "Invalid dim"
    for i in torch.pow(2, torch.arange(math.log2(L)+1, device=v.device, dtype=torch.int64)):
        index = torch.arange(i, L, device=v.device, dtype=torch.int64)
        v.index_copy_(dim, index, ops(v.index_select(dim, index), v.index_select(dim, index-i)))
    return v


def cummul_(v, dim):
    return cumops_(v, dim, lambda a, b : a * b)


def cumprod_(v, dim):
    r'''
        Inplace version of pypose.cumprod
    '''
    return cumops_(v, dim, lambda a, b : a @ b)


def cumops(v, dim, ops):
    return cumops_(v.clone(), dim, ops)


def cummul(v, dim):
    r"""Returns the cumulative multiplication (*) of LieGroup elements of input in the dimension dim.

    For example, if input is a vector of size N, the result will also be a vector of size N, with elements.

    .. math::
        y_i = x_1 * x_2 * \cdots @ x_i

    Args:
        input (LieGroupTensor): the input tenso
        dim (int): the dimension to do the operation over

    Returns:
        LieGroup: The LieGroup Tensor

    Examples:
        >>> input = pp.randn_SE3(2)
        >>> pp.cumprod(input, dim=0)
        SE3Type Group:
        tensor([[-1.9615, -0.1246,  0.3666,  0.0165,  0.2853,  0.3126,  0.9059],
                [ 0.7139,  1.3988, -0.1909, -0.1780,  0.4405, -0.6571,  0.5852]])
    """
    return cumops(v, dim, lambda a, b : a * b)


def cumprod(v, dim):
    r"""Returns the cumulative product (@) of LieGroup elements of input in the dimension dim.

    For example, if input is a vector of size N, the result will also be a vector of size N, with elements.

    .. math::
        y_i = x_1 @ x_2 @ \cdots @ x_i

    Args:
        input (LieGroupTensor): the input tenso
        dim (int): the dimension to do the operation over

    Returns:
        LieGroup: The LieGroup Tensor

    Examples:
        >>> input = pp.randn_SE3(2)
        >>> pp.cumprod(input, dim=0)
        SE3Type Group:
        tensor([[-1.9615, -0.1246,  0.3666,  0.0165,  0.2853,  0.3126,  0.9059],
                [ 0.7139,  1.3988, -0.1909, -0.1780,  0.4405, -0.6571,  0.5852]])
    """
    return cumops(v, dim, lambda a, b : a @ b)
