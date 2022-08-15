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
                        torch.stack([-v[...,1],  v[...,0],         O], dim=-1)], dim=-2)


def add_(input, other, alpha=1):
    r'''
    Inplace version of :meth:`pypose.add`.
    '''
    return input.add_(other, alpha)


def add(input, other, alpha=1):
    r'''
    Adds other, scaled by alpha, to input LieTensor.

    Args:
        input (:obj:`LieTensor`): the input LieTensor (Lie Algebra or Lie Group).

        other (:obj:`Tensor`): the tensor to add to input. The last dimension has to be no less
            than the shape of the corresponding Lie Algebra of the input.

        alpha (:obj:`Number`): the multiplier for other.

    Return:
        :obj:`LieTensor`: the output LieTensor.

    .. math::
        \bm{y}_i =
        \begin{cases}
        \alpha * \bm{a}_i + \bm{x}_i & \text{if}~\bm{x}_i~\text{is a Lie Algebra} \\
        \mathrm{Exp}(\alpha * \bm{a}_i) \times \bm{x}_i & \text{if}~\bm{x}_i~\text{is a Lie Group}
        \end{cases}

    where :math:`\bm{x}` is the ``input`` LieTensor, :math:`\bm{a}` is the ``other`` Tensor to add,
    and :math:`\bm{y}` is the output LieTensor.

    Note:
        A Lie Group normally requires a larger space than its corresponding Lie Algebra, thus
        the elements in the last dimension of the ``other`` Tensor (treated as a Lie Algebra
        in this function) beyond the expected shape of the Lie Algebra are ignored. This is
        because the gradient of a Lie Group is computed as a left perturbation (a Lie Algebra)
        in its tangent space and is stored in the LieGroup's :obj:`LieTensor.grad`, which has
        the same storage space with the LieGroup.

        .. math::
            \begin{align*}
                \frac{D f(\mathcal{X})}{D \mathcal{X}} & \overset{\underset{\mathrm{def}}{}}{=}
                \displaystyle \lim_{\bm{\tau} \to \bm{0}} \frac{f(\bm{\tau} \oplus \mathcal{X})
                    \ominus f(\mathcal{X})}{\bm{\tau}} \\
                & = \left. \frac{\partial \mathrm{Log} (\mathrm{Exp}(\bm{\tau}) \times \mathcal{X})
                    \times f(\mathcal{X})^{-1}}{\partial \bm{\tau}}\right|_{\bm{\tau=\bm{0}}}
            \end{align*},

        where :math:`\mathcal{X}` is a Lie Group and :math:`\bm{\tau}` is its left perturbation.

        See Eq.(44) in `Micro Lie theory <https://arxiv.org/abs/1812.01537>`_ for more details of
        the gradient for a Lie Group.

        This provides convenience to work with PyTorch optimizers like :obj:`torch.optim.SGD`,
        which calls function :meth:`.add_` of a Lie Group to adjust parameters by gradients
        (:obj:`LieTensor.grad`, where the last element is often zero since tangent vector requires
        smaller storage space).

    See :meth:`LieTensor` for types of Lie Algebra and Lie Group.

    See :meth:`Exp` for Exponential mapping of Lie Algebra.

    Examples:
        The following operations are equivalent.

        >>> x = pp.randn_SE3()
        >>> a = torch.randn(6)
        >>> x + a
        SE3Type LieTensor:
        tensor([-1.6089,  0.4184,  0.6621, -0.2098,  0.5383,  0.4794,  0.6606])
        >>> pp.add(x, a)
        SE3Type LieTensor:
        tensor([-1.6089,  0.4184,  0.6621, -0.2098,  0.5383,  0.4794,  0.6606])
        >>> pp.se3(a).Exp() @ x
        SE3Type LieTensor:
        tensor([-1.6089,  0.4184,  0.6621, -0.2098,  0.5383,  0.4794,  0.6606])
        >>> x + torch.cat([a, torch.randn(1)])
        SE3Type LieTensor:
        tensor([-1.6089,  0.4184,  0.6621, -0.2098,  0.5383,  0.4794,  0.6606])
    '''
    return input.add(other, alpha)


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
