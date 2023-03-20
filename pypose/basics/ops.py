import math, torch


def pm(input, *, out=None):
    r'''
    Returns plus or minus (:math:`\pm`) states for tensor.

    Args:
        input (:obj:`Tensor`): the input tensor.
    
    Return:
        :obj:`Tensor`: the output tensor contains only :math:`-1` or :math:`+1`.

    Note:
        The :meth:`pm` function is different from :meth:`torch.sign`, which returns
        :math:`0` for zero inputs, it will return :math:`+1` if an input element is zero.

    Example:
        >>> pp.pm(torch.tensor([0.1, 0, -0.2]))
        tensor([ 1.,  1., -1.])
        >>> pp.pm(torch.tensor([0.1, 0, -0.2], dtype=torch.float64))
        tensor([ 1.,  1., -1.], dtype=torch.float64)
    '''
    out = torch.sign(input, out=None)
    out[out==0] = 1
    return out


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


def cummul(input, dim, left = True):
    r"""Returns the cumulative multiplication (*) of LieTensor along a dimension.

    * Left multiplication:

    .. math::
        y_i = x_i * x_{i-1} * \cdots * x_1,

    * Right multiplication:

    .. math::
        y_i = x_1 * x_2 * \cdots * x_i,
        
    where :math:`x_i,~y_i` are the :math:`i`-th LieType item along the :obj:`dim`
    dimension of input and output, respectively.

    Args:
        input (LieTensor): the input LieTensor
        dim (int): the dimension to do the multiplication over
        left (bool, optional): whether perform left multiplication in :obj:`cummul`.
            If set it to :obj:`False`, this function performs right multiplication.
            Defaul: ``True``

    Returns:
        LieTensor: The LieTensor

    Note:
        - The time complexity of the function is :math:`\mathcal{O}(\log N)`, where
          :math:`N` is the LieTensor size along the :obj:`dim` dimension.

    Example:
    
        * Left multiplication with :math:`\text{input} \in` :obj:`SE3`

        >>> input = pp.randn_SE3(2)
        >>> pp.cummul(input, dim=0)
        SE3Type LieTensor:
        tensor([[-1.9615, -0.1246,  0.3666,  0.0165,  0.2853,  0.3126,  0.9059],
                [ 0.7139,  1.3988, -0.1909, -0.1780,  0.4405, -0.6571,  0.5852]])

        * Left multiplication with :math:`\text{input} \in` :obj:`SO3`

        >>> input = pp.randn_SO3(1,2)
        >>> pp.cummul(input, dim=1, left=False)
        SO3Type LieTensor:
        tensor([[[-1.8252e-01,  1.6198e-01,  8.3683e-01,  4.9007e-01],
                [ 2.0905e-04,  5.2031e-01,  8.4301e-01, -1.3642e-01]]])
    """
    if left:
        return cumops(input, dim, lambda a, b : a * b)
    else: 
        return cumops(input, dim, lambda a, b : b * a)


def cumprod(input, dim, left = True):
    r"""Returns the cumulative product (``@``) of LieTensor along a dimension.

    * Left product:

    .. math::
        y_i = x_i ~\times~ x_{i-1} ~\times~ \cdots ~\times~ x_1,
    
    * Right product:

    .. math::
        y_i = x_1 ~\times~ x_2 ~\times~ \cdots ~\times~ x_i,

    where :math:`\times` denotes the group product (``@``), :math:`x_i,~y_i` are the
    :math:`i`-th item along the :obj:`dim` dimension of the input and output LieTensor,
    respectively.

    Args:
        input (LieTensor): the input LieTensor
        dim (int): the dimension to do the operation over
        left (bool, optional): whether perform left product in :obj:`cumprod`. If set
            it to :obj:`False`, this function performs right product. Defaul: ``True``

    Returns:
        LieTensor: The LieTensor

    Note:
        - The time complexity of the function is :math:`\mathcal{O}(\log N)`, where
          :math:`N` is the LieTensor size along the :obj:`dim` dimension.

    Example:

        * Left product with :math:`\text{input} \in` :obj:`SE3`

        >>> input = pp.randn_SE3(2)
        >>> pp.cumprod(input, dim=0)
        SE3Type LieTensor:
        tensor([[-1.9615, -0.1246,  0.3666,  0.0165,  0.2853,  0.3126,  0.9059],
                [ 0.7139,  1.3988, -0.1909, -0.1780,  0.4405, -0.6571,  0.5852]])

        * Right product with :math:`\text{input} \in` :obj:`SO3`

        >>> input = pp.randn_SO3(1,2)
        >>> pp.cumprod(input, dim=1, left=False)
        SO3Type LieTensor:
        tensor([[[ 0.5798, -0.1189, -0.2429,  0.7686],
                [ 0.7515, -0.1920,  0.5072,  0.3758]]])
    """
    if left:
        return cumops(input, dim, lambda a, b : a @ b)
    else:
        return cumops(input, dim, lambda a, b : b @ a)
