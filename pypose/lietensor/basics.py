import math
import torch


def vec2skew(input:torch.Tensor) -> torch.Tensor:
    r"""
    Convert batched vectors to skew matrices.

    Args:
        input (Tensor): the tensor :math:`\mathbf{x}` to convert.

    Return:
        Tensor: the skew matrices :math:`\mathbf{y}`.

    Shape:
        Input: :obj:`(*, 3)`

        Output: :obj:`(*, 3, 3)`

    .. math::
        {\displaystyle \mathbf{y}_i={\begin{bmatrix}\,\,
        0&\!-x_{i,3}&\,\,\,x_{i,2}\\\,\,\,x_{i,3}&0&\!-x_{i,1}
        \\\!-x_{i,2}&\,\,x_{i,1}&\,\,0\end{bmatrix}}}

    Note:
        The last dimension of the input tensor has to be 3.

    Example:
        >>> pp.vec2skew(torch.randn(1,3))
        tensor([[[ 0.0000, -2.2059, -1.2761],
                [ 2.2059,  0.0000,  0.2929],
                [ 1.2761, -0.2929,  0.0000]]])
    """
    v = input.tensor() if hasattr(input, 'ltype') else input
    assert v.shape[-1] == 3, "Last dim should be 3"
    O = torch.zeros(v.shape[:-1], device=v.device, dtype=v.dtype, requires_grad=v.requires_grad)
    return torch.stack([torch.stack([        O, -v[...,2],  v[...,1]], dim=-1),
                        torch.stack([ v[...,2],         O, -v[...,0]], dim=-1),
                        torch.stack([-v[...,1],  v[...,0],         O], dim=-1)], dim=-1)


def cumops_(input, dim, ops):
    r'''
        Inplace version of :meth:`pypose.cumops`
    '''
    L, v = input.shape[dim], input
    assert dim != -1 or dim != v.shape[-1], "Invalid dim"
    for i in torch.pow(2, torch.arange(math.log2(L)+1, device=v.device, dtype=torch.int64)):
        index = torch.arange(i, L, device=v.device, dtype=torch.int64)
        v.index_copy_(dim, index, ops(v.index_select(dim, index), v.index_select(dim, index-i)))
    return v


def cummul_(input, dim):
    r'''
        Inplace version of :meth:`pypose.cummul`
    '''
    return cumops_(input, dim, lambda a, b : a * b)


def cumprod_(input, dim):
    r'''
        Inplace version of :meth:`pypose.cumprod`
    '''
    return cumops_(input, dim, lambda a, b : a @ b)


def cumops(input, dim, ops):
    r"""Returns the cumulative user-defined operation of LieTensor along a dimension.

    .. math::
        y_i = x_1~\mathrm{\circ}~x_2 ~\mathrm{\circ}~ \cdots ~\mathrm{\circ}~ x_i,

    where :math:`\mathrm{\circ}` is the user-defined operation and :math:`x_i,~y_i`
    are the :math:`i`-th LieType item along the :obj:`dim` dimension of input and
    output, respectively.

    Args:
        input (LieTensor): the input LieTensor
        dim (int): the dimension to do the operation over
        ops (func): the user-defined operation or function

    Returns:
        LieTensor: LieTensor

    Note:
        - The users are supposed to provide meaningful operation.
        - This function doesn't check whether the results are valid for mathematical
          definition of LieTensor, e.g., quaternion.
        - The time complexity of the function is :math:`\mathcal{O}(\log N)`, where
          :math:`N` is the LieTensor size along the :obj:`dim` dimension.

    Examples:
        >>> input = pp.randn_SE3(2)
        >>> input.cumprod(dim = 0)
        SE3Type LieTensor:
        tensor([[-0.6466,  0.2956,  2.4055, -0.4428,  0.1893,  0.3933,  0.7833],
                [ 1.2711,  1.2020,  0.0651, -0.0685,  0.6732,  0.7331, -0.0685]])
        >>> pp.cumops(input, 0, lambda a, b : a @ b)
        SE3Type LieTensor:
        tensor([[-0.6466,  0.2956,  2.4055, -0.4428,  0.1893,  0.3933,  0.7833],
                [ 1.2711,  1.2020,  0.0651, -0.0685,  0.6732,  0.7331, -0.0685]])
    """
    return cumops_(input.clone(), dim, ops)


def cummul(input, dim):
    r"""Returns the cumulative multiplication (*) of LieTensor along a dimension.

    .. math::
        y_i = x_1 * x_2 * \cdots * x_i,

    where :math:`x_i,~y_i` are the :math:`i`-th LieType item along the :obj:`dim`
    dimension of input and output, respectively.

    Args:
        input (LieTensor): the input LieTensor
        dim (int): the dimension to do the multiplication over

    Returns:
        LieTensor: The LieTensor

    Note:
        - The time complexity of the function is :math:`\mathcal{O}(\log N)`, where
          :math:`N` is the LieTensor size along the :obj:`dim` dimension.

    Examples:
        >>> input = pp.randn_SE3(2)
        >>> pp.cumprod(input, dim=0)
        SE3Type LieTensor:
        tensor([[-1.9615, -0.1246,  0.3666,  0.0165,  0.2853,  0.3126,  0.9059],
                [ 0.7139,  1.3988, -0.1909, -0.1780,  0.4405, -0.6571,  0.5852]])
    """
    return cumops(input, dim, lambda a, b : a * b)


def cumprod(input, dim):
    r"""Returns the cumulative product (@) of LieTensor along a dimension.

    .. math::
        y_i = x_1 ~@~ x_2 ~@~ \cdots ~@~ x_i,

    where :math:`x_i,~y_i` are the :math:`i`-th LieType item along the :obj:`dim`
    dimension of input and output, respectively.

    Args:
        input (LieTensor): the input LieTensor
        dim (int): the dimension to do the operation over

    Returns:
        LieTensor: The LieTensor

    Note:
        - The time complexity of the function is :math:`\mathcal{O}(\log N)`, where
          :math:`N` is the LieTensor size along the :obj:`dim` dimension.

    Examples:
        >>> input = pp.randn_SE3(2)
        >>> pp.cumprod(input, dim=0)
        SE3Type LieTensor:
        tensor([[-1.9615, -0.1246,  0.3666,  0.0165,  0.2853,  0.3126,  0.9059],
                [ 0.7139,  1.3988, -0.1909, -0.1780,  0.4405, -0.6571,  0.5852]])
    """
    return cumops(input, dim, lambda a, b : a @ b)
