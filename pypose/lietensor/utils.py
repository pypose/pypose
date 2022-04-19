import torch
import functools
from .lietensor import  LieTensor
from .lietensor import SE3_type, se3_type
from .lietensor import SO3_type, so3_type
from .lietensor import Sim3_type, sim3_type
from .lietensor import RxSO3_type, rxso3_type


SO3 = functools.partial(LieTensor, ltype=SO3_type)
SO3.__doc__ = r'''
Alias of LieTensor for SO3.
'''

so3 = functools.partial(LieTensor, ltype=so3_type)
so3.__doc__ = r'''
Alias of LieTensor for so3.
'''

SE3 = functools.partial(LieTensor, ltype=SE3_type)
SE3.__doc__ = r'''
Alias of LieTensor for SE3.
'''

se3 = functools.partial(LieTensor, ltype=se3_type)
se3.__doc__ = r'''
Alias of LieTensor for se3.
'''

Sim3 = functools.partial(LieTensor, ltype=Sim3_type)
Sim3.__doc__ = r'''
Alias of LieTensor for Sim3.
'''

sim3 = functools.partial(LieTensor, ltype=sim3_type)
sim3.__doc__ = r'''
Alias of LieTensor for sim3.
'''

RxSO3 = functools.partial(LieTensor, ltype=RxSO3_type)
RxSO3.__doc__ = r'''
Alias of LieTensor for RxSO3.
'''

rxso3 = functools.partial(LieTensor, ltype=rxso3_type)
rxso3.__doc__ = r'''
Alias of LieTensor for rxso3.
'''

def randn_like(input, sigma=1, **kwargs):
    r'''
    Returns a LieTensor with the same size as input that is filled with random
    LieTensor that satisfies the corresponding :obj:`input.ltype`.

    The corresponding random generator can be

    .. list-table:: List of available random LieTensor generator of input :obj:`ltype`.
        :widths: 25 25 30 30 30
        :header-rows: 1

        * - Name
          - ltype
          - randn function
          - Manifold
          - randn function
        * - Orthogonal Group
          - :obj:`SO3_type`
          - :meth:`randn_SO3`
          - :obj:`so3_type`
          - :meth:`randn_so3`
        * - Euclidean Group
          - :obj:`SE3_type`
          - :meth:`randn_SE3`
          - :obj:`se3_type`
          - :meth:`randn_se3`
        * - Similarity Group
          - :obj:`Sim3_type`
          - :meth:`randn_Sim3`
          - :obj:`sim3_type`
          - :meth:`randn_sim3`
        * - Scaling Orthogonal
          - :obj:`RxSO3_type`
          - :meth:`randn_RxSO3`
          - :obj:`rxso3_type`
          - :meth:`randn_rxso3`

    Args:

        input (LieTensor): the size of input will determine size of the output tensor.

        dtype (torch.dtype, optional): the desired data type of returned Tensor.
            Default: if None, defaults to the dtype of input.

        layout (torch.layout, optional): the desired layout of returned tensor.
            Default: if None, defaults to the layout of input.

        device (torch.device, optional): the desired device of returned tensor.
            Default: if None, defaults to the device of input.

        requires_grad (bool, optional): If autograd should record operations on the returned tensor.
            Default: False.

        memory_format (torch.memory_format, optional): the desired memory format of returned Tensor.
            Default: torch.preserve_format.

    Note:
        If we have:

        .. code::

            import pypose as pp
            x = pp.SO3(data)

        Then the following two usages are equivalent:

        .. code::

            pp.randn_like(x)
            pp.randn_SO3(x.lshape, dtype=x.dtype, layout=x.layout, device=x.device)

    Example:
        >>> x = pp.so3(torch.tensor([0, 0, 0]))
        >>> pp.randn_like(x)
        so3Type LieTensor:
        tensor([0.8970, 0.0943, 0.1399])
    '''
    return input.ltype.randn_like(*input.lshape, sigma=sigma, **kwargs)


def randn_so3(*size, sigma=1, **kwargs):
    r'''
    Returns :obj:`so3_type` LieTensor filled with random numbers from a normal
    distribution with mean 0 and variance :obj:`sigma` (also called the standard normal distribution).

    .. math::
        \mathrm{out}_i = \mathcal{N}(\mathbf{0}_{3\times 1}, \mathbf{\sigma}_{3\times 1})

    The shape of the tensor is defined by the variable argument size.

    Args:
        size (int...): a sequence of integers defining the shape of the output tensor.
            Can be a variable number of arguments or a collection like a list or tuple.

        sigma (float, optional): variance of the normal distribution. Default: 1.

        requires_grad (bool, optional): If autograd should record operations on
            the returned tensor. Default: False.

        generator (torch.Generator, optional): a pseudorandom number generator for sampling

        dtype (torch.dtype, optional): the desired data type of returned tensor.
            Default: if None, uses a global default (see :meth:`torch.set_default_tensor_type()`).

        layout (torch.layout, optional): the desired layout of returned Tensor.
            Default: torch.strided.

        device (torch.device, optional): the desired device of returned tensor.
            Default: if None, uses the current device for the default tensor
            type (see :meth:`torch.set_default_tensor_type()`). device will be the CPU
            for CPU tensor types and the current CUDA device for CUDA tensor types.

    Returns:
        LieTensor: a :obj:`so3_type` LieTensor

    Example:
        >>> pp.randn_so3(2, sigma=0.1, requires_grad=True, dtype=torch.float64)
        so3Type LieTensor:
        tensor([[-0.0427, -0.0149,  0.0948],
                [ 0.0607,  0.0473,  0.0703]], dtype=torch.float64, requires_grad=True)
    '''
    return so3_type.randn(*size, sigma=sigma, **kwargs)


def randn_SO3(*size, sigma=1, **kwargs):
    r'''
    Returns :obj:`SO3_type` LieTensor filled with the Exponential map of the random
    :obj:`so3_type` LieTensor with normal distribution with mean 0 and variance :obj:`sigma`.

    .. math::
        \mathrm{out}_i = \mathrm{Exp}(\mathcal{N}(\mathbf{0}_{3\times 1}, \mathbf{\sigma}_{3\times 1}))

    The shape of the tensor is defined by the variable argument size.

    Args:
        size (int...): a sequence of integers defining the shape of the output tensor.
            Can be a variable number of arguments or a collection like a list or tuple.

        sigma (float, optional): variance :math:`\sigma` of the normal distribution. Default: 1.

        requires_grad (bool, optional): If autograd should record operations on
            the returned tensor. Default: False.

        generator (torch.Generator, optional): a pseudorandom number generator for sampling

        dtype (torch.dtype, optional): the desired data type of returned tensor.
            Default: if None, uses a global default (see :meth:`torch.set_default_tensor_type()`).

        layout (torch.layout, optional): the desired layout of returned Tensor.
            Default: torch.strided.

        device (torch.device, optional): the desired device of returned tensor.
            Default: if None, uses the current device for the default tensor
            type (see :meth:`torch.set_default_tensor_type()`). device will be the CPU
            for CPU tensor types and the current CUDA device for CUDA tensor types.

    Returns:
        LieTensor: a :obj:`SO3_type` LieTensor

    Example:
        >>> pp.randn_SO3(2, sigma=0.1, requires_grad=True, dtype=torch.float64)
        SO3Type LieTensor:
        tensor([[-0.0060, -0.0517, -0.0070,  0.9986],
                [ 0.0015,  0.0753,  0.0503,  0.9959]], dtype=torch.float64, requires_grad=True)

    '''
    return SO3_type.randn(*size, sigma=sigma, **kwargs)


def randn_se3(*size, sigma=1, **kwargs):
    r'''
    Returns :obj:`se3_type` LieTensor filled with random numbers from a normal
    distribution with mean 0 and variance :obj:`sigma` (also called the standard normal distribution).

    .. math::
        \mathrm{out}_i = \mathcal{N}(\mathbf{0}_{6\times 1}, \mathbf{\sigma}_{6\times 1})

    The shape of the tensor is defined by the variable argument size.

    Args:
        size (int...): a sequence of integers defining the shape of the output tensor.
            Can be a variable number of arguments or a collection like a list or tuple.

        sigma (float, optional): variance of the normal distribution. Default: 1.

        requires_grad (bool, optional): If autograd should record operations on
            the returned tensor. Default: False.

        generator (torch.Generator, optional): a pseudorandom number generator for sampling

        dtype (torch.dtype, optional): the desired data type of returned tensor.
            Default: if None, uses a global default (see :meth:`torch.set_default_tensor_type()`).

        layout (torch.layout, optional): the desired layout of returned Tensor.
            Default: torch.strided.

        device (torch.device, optional): the desired device of returned tensor.
            Default: if None, uses the current device for the default tensor
            type (see :meth:`torch.set_default_tensor_type()`). device will be the CPU
            for CPU tensor types and the current CUDA device for CUDA tensor types.

    Returns:
        LieTensor: a :obj:`se3_type` LieTensor

    Example:
        >>> pp.randn_se3(2, sigma=0.1, requires_grad=True, dtype=torch.float64)
        se3Type LieTensor:
        tensor([[-0.0599, -0.0593,  0.0809,  0.0352, -0.2173,  0.0342],
                [-0.0226, -0.1081,  0.0270,  0.1368, -0.0327, -0.2052]],
            dtype=torch.float64, requires_grad=True)
    '''
    return se3_type.randn(*size, sigma=sigma, **kwargs)


def randn_SE3(*size, sigma=1, **kwargs):
    r'''
    Returns :obj:`SE3_type` LieTensor filled with the Exponential map of the random
    :obj:`se3_type` LieTensor with normal distribution with mean 0 and variance :obj:`sigma`.

    .. math::
        \mathrm{out}_i = \mathrm{Exp}(\mathcal{N}(\mathbf{0}_{6\times 1}, \mathbf{\sigma}_{6\times 1}))

    The shape of the tensor is defined by the variable argument size.

    Args:
        size (int...): a sequence of integers defining the shape of the output tensor.
            Can be a variable number of arguments or a collection like a list or tuple.

        sigma (float, optional): variance :math:`\sigma` of the normal distribution. Default: 1.

        requires_grad (bool, optional): If autograd should record operations on
            the returned tensor. Default: False.

        generator (torch.Generator, optional): a pseudorandom number generator for sampling

        dtype (torch.dtype, optional): the desired data type of returned tensor.
            Default: if None, uses a global default (see :meth:`torch.set_default_tensor_type()`).

        layout (torch.layout, optional): the desired layout of returned Tensor.
            Default: torch.strided.

        device (torch.device, optional): the desired device of returned tensor.
            Default: if None, uses the current device for the default tensor
            type (see :meth:`torch.set_default_tensor_type()`). device will be the CPU
            for CPU tensor types and the current CUDA device for CUDA tensor types.

    Returns:
        LieTensor: a :obj:`SE3_type` LieTensor

    Example:
        >>> pp.randn_SE3(2, sigma=0.1)
        SE3Type LieTensor:
        tensor([[-0.0522, -0.0456, -0.1996,  0.0266, -0.0240, -0.0375,  0.9987],
                [-0.1344, -0.1673,  0.1111, -0.0219, -0.0454,  0.0710,  0.9962]])
    '''
    return SE3_type.randn(*size, sigma=sigma, **kwargs)


def randn_sim3(*size, sigma=1, **kwargs):
    return sim3_type.randn(*size, sigma=sigma, **kwargs)


def randn_Sim3(*size, sigma=1, **kwargs):
    return Sim3_type.randn(*size, sigma=sigma, **kwargs)


def randn_rxso3(*size, sigma=1, **kwargs):
    return rxso3_type.randn(*size, sigma=sigma, **kwargs)


def randn_RxSO3(*size, sigma=1, **kwargs):
    return RxSO3_type.randn(*size, sigma=sigma, **kwargs)


def identity_like(liegroup, **kwargs):
    return liegroup.ltype.identity_like(*liegroup.lshape, **kwargs)


def identity_SO3(*size, **kwargs):
    return SO3_type.identity(*size, **kwargs)


def identity_so3(*size, **kwargs):
    return so3_type.identity(*size, **kwargs)


def identity_SE3(*size, **kwargs):
    return SE3_type.identity(*size, **kwargs)


def identity_se3(*size, **kwargs):
    return se3_type.identity(*size, **kwargs)


def identity_sim3(*size, **kwargs):
    return sim3_type.identity(*size, **kwargs)


def identity_Sim3(*size, **kwargs):
    return Sim3_type.identity(*size, **kwargs)


def identity_rxso3(*size, **kwargs):
    return rxso3_type.identity(*size, **kwargs)


def identity_RxSO3(*size, **kwargs):
    return RxSO3_type.identity(*size, **kwargs)


def assert_ltype(func):
    @functools.wraps(func)
    def checker(*args, **kwargs):
        assert isinstance(args[0], LieTensor), "Invalid LieTensor Type."
        out = func(*args, **kwargs)
        return out
    return checker


@assert_ltype
def Exp(input):
    r"""The Exponential map for :obj:`LieTensor` (Lie Algebra).

    .. math::
        \mathrm{Exp}: \mathcal{g} \mapsto \mathcal{G}

    Args:
        input (LieTensor): the input LieTensor (Lie Algebra)

    Return:
        LieTensor: the output LieTensor (Lie Group)

    .. list-table:: List of supported :math:`\mathrm{Exp}` map
        :widths: 20 20 8 20 20
        :header-rows: 1

        * - input :obj:`ltype`
          - :math:`\mathcal{g}` (Lie Algebra)
          - :math:`\mapsto`
          - :math:`\mathcal{G}` (Lie Group)
          - output :obj:`ltype`
        * - :obj:`so3_type`
          - :math:`\mathcal{g}\in\mathbb{R}^{*\times3}`
          - :math:`\mapsto`
          - :math:`\mathcal{G}\in\mathbb{R}^{*\times4}`
          - :obj:`SO3_type`
        * - :obj:`se3_type`
          - :math:`\mathcal{g}\in\mathbb{R}^{*\times6}`
          - :math:`\mapsto`
          - :math:`\mathcal{G}\in\mathbb{R}^{*\times7}`
          - :obj:`SE3_type`
        * - :obj:`sim3_type`
          - :math:`\mathcal{g}\in\mathbb{R}^{*\times7}`
          - :math:`\mapsto`
          - :math:`\mathcal{G}\in\mathbb{R}^{*\times8}`
          - :obj:`Sim3_type`
        * - :obj:`rxso3_type`
          - :math:`\mathcal{g}\in\mathbb{R}^{*\times4}`
          - :math:`\mapsto`
          - :math:`\mathcal{G}\in\mathbb{R}^{*\times5}`
          - :obj:`RxSO3_type`

    Note:
        This function :func:`Exp()` is different from :func:`exp()`, which returns
        a new torch tensor with the exponential of the elements of the input tensor.

    * Input :math:`\mathbf{x}`'s :obj:`ltype` is :obj:`so3_type`
      (input :math:`\mathbf{x}` is an instance of :meth:`so3`):

        If :math:`\|\mathbf{x}_i\| > \text{eps}`:

        .. math::
            \mathbf{y}_i = \left[\mathbf{x}_{i,1}\theta_i,
            \mathbf{x}_{i,2}\theta_i,
            \mathbf{x}_{i,3}\theta_i,
            \cos(\frac{\|\mathbf{x}_i\|}{2})\right],

        where :math:`\theta_i = \frac{1}{\|\mathbf{x}_i\|}\sin(\frac{\|\mathbf{x}_i\|}{2})`,

        otherwise:

        .. math::
            \mathbf{y}_i = \left[\mathbf{x}_{i,1}\theta_i,~
            \mathbf{x}_{i,2}\theta_i,~
            \mathbf{x}_{i,3}\theta_i,~
            1 - \frac{\|\mathbf{x}_i\|^2}{8} + \frac{\|\mathbf{x}_i\|^4}{384} \right],

        where :math:`\theta_i = \frac{1}{2} - \frac{1}{48} \|\mathbf{x}_i\|^2 + \frac{1}{3840} \|\mathbf{x}_i\|^4`.

    * Input :math:`\mathbf{x}`'s :obj:`ltype` is :obj:`se3_type`
      (input :math:`\mathbf{x}` is an instance of :meth:`se3`):

        Let :math:`\bm{\tau}_i`, :math:`\bm{\phi}_i` be the translation and rotation parts of
        :math:`\mathbf{x}_i`, respectively; :math:`\mathbf{y}` be the output.

        .. math::
            \mathbf{y}_i = \left[\mathbf{J}_i\bm{\tau}_i, \mathrm{Exp}(\bm{\phi}_i)\right],
        
        where :math:`\mathrm{Exp}` is the Exponential map for :obj:`so3_type` input and
        :math:`\mathbf{J}_i` is the left Jacobian.

    * Input :math:`\mathbf{x}`'s :obj:`ltype` is :obj:`rxso3_type`
      (input :math:`\mathbf{x}` is an instance of :meth:`rxso3`):

        Let :math:`\bm{\phi}_i`, :math:`s_i` be the rotation and scale parts of
        :math:`\mathbf{x}_i`, respectively; :math:`\mathbf{y}` be the output.

        .. math::
            \mathbf{y}_i = \left[\mathrm{Exp}(\bm{\phi}_i), \mathrm{exp}(s_i)\right],

        where :math:`\mathrm{exp}` is the exponential function.

    * Input :math:`\mathbf{x}`'s :obj:`ltype` is :obj:`sim3_type`
      (input :math:`\mathbf{x}` is an instance of :meth:`sim3`):

        Let :math:`\bm{\tau}_i`, :math:`^{s}\bm{\phi}_i` be the translation and
        :meth:`rxso3` parts of :math:`\mathbf{x}_i`, respectively; :math:`\mathbf{y}` be the output.

        .. math::
            \mathbf{y}_i = \left[^{s}\mathbf{J}_i\bm{\tau}_i, \mathrm{Exp}(^{s}\bm{\phi}_i)\right],
        
        where :math:`^{s}\mathbf{J}` is the similarity transformed left Jacobian.

    Example:
    
    * :math:`\mathrm{Exp}`: :meth:`so3` :math:`\mapsto` :meth:`SO3`

        >>> x = pp.randn_so3(2, requires_grad=True)
        so3Type LieTensor:
        tensor([[-0.2547, -0.4478,  0.0783],
                [ 0.7381,  0.2163, -1.8465]], requires_grad=True)
        >>> x.Exp() # equivalent to: pp.Exp(x)
        SO3Type LieTensor:
        tensor([[-0.1259, -0.2214,  0.0387,  0.9662],
                [ 0.3105,  0.0910, -0.7769,  0.5402]], grad_fn=<AliasBackward0>)

    * :math:`\mathrm{Exp}`: :meth:`se3` :math:`\mapsto` :meth:`SE3`

        >>> x = pp.randn_se3(2)
        se3Type LieTensor:
        tensor([[ 1.1912,  1.2425, -0.9696,  0.9540, -0.4061, -0.7204],
                [ 0.5964, -1.1894,  0.6451,  1.1373, -2.6733,  0.4142]])
        >>> x.Exp() # equivalent to: pp.Exp(x)
        SE3Type LieTensor:
        tensor([[ 1.6575,  0.8838, -0.1499,  0.4459, -0.1898, -0.3367,  0.8073],
                [ 0.2654, -1.3860,  0.2852,  0.3855, -0.9061,  0.1404,  0.1034]])

    * :math:`\mathrm{Exp}`: :meth:`rxso3` :math:`\mapsto` :meth:`RxSO3`

        >>> x = pp.randn_rxso3(2)
        rxso3Type LieTensor:
        tensor([[-1.2559, -0.9545,  0.2480, -0.3000],
                [ 1.0867,  0.4305, -0.4303,  0.1563]])
        >>> x.Exp() # equivalent to: pp.Exp(x)
        RxSO3Type LieTensor:
        tensor([[-0.5633, -0.4281,  0.1112,  0.6979,  0.7408],
                [ 0.5089,  0.2016, -0.2015,  0.8122,  1.1692]])

    * :math:`\mathrm{Exp}`: :meth:`sim3` :math:`\mapsto` :meth:`Sim3`

        >>> x = pp.randn_sim3(2)
        sim3Type LieTensor:
        tensor([[-1.2279,  0.0967, -1.1261,  1.2900,  0.2519, -0.7583,  0.8938],
                [ 0.4278, -0.4025, -1.3189, -1.7345, -0.9196,  0.3332,  0.1777]])
        >>> x.Exp() # equivalent to: pp.Exp(x)
        Sim3Type LieTensor:
        tensor([[-1.5811,  1.8128, -0.5835,  0.5849,  0.1142, -0.3438,  0.7257,  2.4443],
                [ 0.9574, -0.9265, -0.2385, -0.7309, -0.3875,  0.1404,  0.5440,  1.1945]])
    """
    return input.Exp()


@assert_ltype
def Log(input):
    r"""The Logarithm map for :obj:`LieTensor` (Lie Group).

    .. math::
        \mathrm{Log}: \mathcal{G} \mapsto \mathcal{g}

    Args:
        input (LieTensor): the input LieTensor (Lie Group)

    Return:
        LieTensor: the output LieTensor (Lie Algebra)

    .. list-table:: List of supported :math:`\mathrm{Log}` map
        :widths: 20 20 8 20 20
        :header-rows: 1

        * - input :obj:`ltype`
          - :math:`\mathcal{G}` (Lie Group)
          - :math:`\mapsto`
          - :math:`\mathcal{g}` (Lie Algebra)
          - output :obj:`ltype`
        * - :obj:`SO3_type`
          - :math:`\mathcal{G}\in\mathbb{R}^{*\times4}`
          - :math:`\mapsto`
          - :math:`\mathcal{g}\in\mathbb{R}^{*\times3}`
          - :obj:`so3_type`
        * - :obj:`SE3_type`
          - :math:`\mathcal{G}\in\mathbb{R}^{*\times7}`
          - :math:`\mapsto`
          - :math:`\mathcal{g}\in\mathbb{R}^{*\times6}`
          - :obj:`se3_type`
        * - :obj:`Sim3_type`
          - :math:`\mathcal{G}\in\mathbb{R}^{*\times8}`
          - :math:`\mapsto`
          - :math:`\mathcal{g}\in\mathbb{R}^{*\times7}`
          - :obj:`sim3_type`
        * - :obj:`RxSO3_type`
          - :math:`\mathcal{G}\in\mathbb{R}^{*\times5}`
          - :math:`\mapsto`
          - :math:`\mathcal{g}\in\mathbb{R}^{*\times4}`
          - :obj:`rxso3_type`
    
    Warning:
        This function :func:`Log()` is different from :func:`log()`, which returns
        a new torch tensor with the logarithm of the elements of the input tensor.

    * If input :math:`\mathbf{x}`'s :obj:`ltype` is :obj:`SO3_type`
      (input :math:`\mathbf{x}` is an instance of :meth:`SO3`):

        Let :math:`w_i`, :math:`\boldsymbol{\nu}_i` be the scalar and vector parts of
        :math:`\mathbf{x}_i`, respectively; :math:`\mathbf{y}` be the output.

        If :math:`\|\boldsymbol{\nu}_i\| > \text{eps}`:

            .. math::
                \mathbf{y}_i = \left\{
                                \begin{array}{ll} 
                                    2\frac{\mathrm{arctan}(\|\boldsymbol{\nu}_i\|/w_i)}{\|
                                    \boldsymbol{\nu}_i\|}\boldsymbol{\nu}_i, \quad \|w_i\| > \text{eps}, \\
                                    \mathrm{sign}(w_i) \frac{\pi}{\|\boldsymbol{\nu}_i\|}\boldsymbol{\nu}_i,
                                    \quad \|w_i\| \leq \text{eps},
                                \end{array}
                             \right.

        otherwise:

        .. math::
            \mathbf{y}_i = 2\left( \frac{1}{w_i} - \frac{\|\boldsymbol{\nu}_i\|^2}{3w_i^3}\right)\boldsymbol{\nu}_i.

    * If input :math:`\mathbf{x}`'s :obj:`ltype` is :obj:`SE3_type`
      (input :math:`\mathbf{x}` is an instance of :meth:`SE3`):

        Let :math:`\mathbf{t}_i`, :math:`\mathbf{q}_i` be the translation and rotation parts of
        :math:`\mathbf{x}_i`, respectively; :math:`\mathbf{y}` be the output.

        .. math::
            \mathbf{y}_i = \left[\mathbf{J}_i^{-1}\mathbf{t}_i, \mathrm{Log}(\mathbf{q}_i) \right],

        where :math:`\mathrm{Log}` is the Logarithm map for :obj:`SO3_type` input and
        :math:`\mathbf{J}_i` is the left Jacobian.

    * If input :math:`\mathbf{x}`'s :obj:`ltype` is :obj:`RxSO3_type`
      (input :math:`\mathbf{x}` is an instance of :meth:`RxSO3`):

        Let :math:`\mathbf{q}_i`, :math:`s_i` be the rotation and scale parts of :math:`\mathbf{x}_i`, respectively;
        :math:`\mathbf{y}` be the output.

        .. math::
            \mathbf{y}_i = \left[\mathrm{Log}(\mathbf{q}_i), \log(s_i) \right].

    * If input :math:`\mathbf{x}`'s :obj:`ltype` is :obj:`Sim3_type` (input :math:`\mathbf{x}`
      is an instance of :meth:`Sim3`):

        Let :math:`\mathbf{t}_i`, :math:`^s\mathbf{q}_i` be the translation and :obj:`RxSO3` parts
        of :math:`\mathbf{x}_i`, respectively; :math:`\mathbf{y}` be the output.

        .. math::
            \mathbf{y}_i = \left[^s\mathbf{J}_i^{-1}\mathbf{t}_i, \mathrm{Log}(^s\mathbf{q}_i) \right],

        where :math:`^s\mathbf{J}_i` is the similarity transformed left Jacobian.

    Note:
        The :math:`\mathrm{arctan}`-based Logarithm map implementation thanks to the paper:

        * C. Hertzberg et al., `Integrating Generic Sensor Fusion Algorithms with Sound State
          Representation through Encapsulation of Manifolds <https://doi.org/10.1016/j.inffus.2011.08.003>`_,
          Information Fusion, 2013.

        Assume we have a unit rotation axis :math:`\mathbf{n}` and rotation angle :math:`\theta~(0\leq\theta<2\pi)`, then
        the corresponding quaternion with unit norm :math:`\mathbf{q}` can be represented as

            .. math::
                \mathbf{q} = \left[\sin(\theta/2) \mathbf{n}, \cos(\theta/2) \right]

        Therefore, given a quaternion :math:`\mathbf{q}=[\boldsymbol{\nu}, w]`, where :math:`\boldsymbol{\nu}` is the vector part,
        :math:`w` is the scalar part, to find the corresponding rotation vector , the rotation angle :math:`\theta` can be obtained as 

            .. math::
                \theta = 2\mathrm{arctan}(\|\boldsymbol{\nu}\|/w),~\|\boldsymbol{\nu}\| = \sin(\theta/2), 

        The unit rotation axis :math:`\mathbf{n}` can be obtained as :math:`\mathbf{n} = \frac{\boldsymbol{\nu}}{{\|\boldsymbol{\nu}\|}}`.     
        Hence, the corresponding rotation vector is 
        
            .. math::
                \theta \mathbf{n} = 2\frac{\mathrm{arctan}(\|\boldsymbol{\nu}\|/w)}{\|\boldsymbol{\nu}\|}\boldsymbol{\nu}.

    Example:

        * :math:`\mathrm{Log}`: :obj:`SO3` :math:`\mapsto` :obj:`so3`

        >>> x = pp.randn_SO3(2)
        >>> x.Log() # equivalent to: pp.Log(x)
        so3Type LieTensor:
        tensor([[-0.3060,  0.2344,  1.2724],
                [ 0.3012, -0.6817,  0.1187]])

        * :math:`\mathrm{Log}`: :obj:`SE3` :math:`\mapsto` :obj:`se3`

        >>> x = pp.randn_SE3(2)
        >>> x.Log() # equivalent to: pp.Log(x)
        se3Type LieTensor:
        tensor([[ 0.2958, -0.0840, -1.4733,  0.7004,  0.4483, -0.9009],
                [ 0.0850, -0.1020, -1.2616, -1.0524, -1.2031,  0.8377]])


        * :math:`\mathrm{Log}`: :obj:`RxSO3` :math:`\mapsto` :obj:`rxso3`

        >>> x = pp.randn_RxSO3(2)
        >>> x.Log() # equivalent to: pp.Log(x)
        rxso3Type LieTensor:
        tensor([[-1.3755,  0.3525, -2.2367,  0.5409],
                [ 0.5929, -0.3250, -0.7394,  1.0965]])

        * :math:`\mathrm{Log}`: :obj:`Sim3` :math:`\mapsto` :obj:`sim3`

        >>> x = pp.randn_Sim3(2)
        >>> x.Log() # equivalent to: pp.Log(x)
        sim3Type LieTensor:
        tensor([[-0.1747, -0.3698,  0.2000,  0.1735,  0.6220,  1.1852, -0.6402],
                [-0.8685, -0.1717,  1.2139, -0.8385, -2.2957, -1.9545,  0.8474]])
    """
    return input.Log()


@assert_ltype
def Inv(x):
    return x.Inv()


@assert_ltype
def Mul(x, y):
    return x * y


@assert_ltype
def Retr(X, a):
    return X.Retr(a)


@assert_ltype
def Act(X, p):
    return X.Act(p)


@assert_ltype
def Adj(X, a):
    return X.Adj(a)


@assert_ltype
def AdjT(X, a):
    return X.AdjT(a)


@assert_ltype
def Jinvp(input, p):
    r"""
    The dot product between left Jacobian inverse at the point given
    by input (Lie Group) and second point (Lie Algebra).

    .. math::
        \mathrm{Jinvp}: (\mathcal{G}, \mathcal{g}) \mapsto \mathcal{g}

    Args:
        input (LieTensor): the input LieTensor (Lie Group)
        p (LieTensor): the second LieTensor (Lie Algebra)

    Return:
        LieTensor: the output LieTensor (Lie Algebra)

    .. list-table:: List of supported :math:`\mathrm{Jinvp}` map
        :widths: 20 20 8 20 20
        :header-rows: 1

        * - input :obj:`ltype`
          - :math:`(\mathcal{G}, \mathcal{g})` (Lie Group, Lie Algebra)
          - :math:`\mapsto`
          - :math:`\mathcal{g}` (Lie Algebra)
          - output :obj:`ltype`
        * - (:obj:`SO3_type`, :obj:`so3_type`)
          - :math:`(\mathcal{G}\in\mathbb{R}^{*\times4}, \mathcal{g}\in\mathbb{R}^{*\times3})`
          - :math:`\mapsto`
          - :math:`\mathcal{g}\in\mathbb{R}^{*\times3}`
          - :obj:`so3_type`
        * - (:obj:`SE3_type`, :obj:`se3_type`)
          - :math:`(\mathcal{G}\in\mathbb{R}^{*\times7}, \mathcal{g}\in\mathbb{R}^{*\times6})`
          - :math:`\mapsto`
          - :math:`\mathcal{g}\in\mathbb{R}^{*\times6}`
          - :obj:`se3_type`
        * - (:obj:`Sim3_type`, :obj:`sim3_type`)
          - :math:`(\mathcal{G}\in\mathbb{R}^{*\times8}, \mathcal{g}\in\mathbb{R}^{*\times7})`
          - :math:`\mapsto`
          - :math:`\mathcal{g}\in\mathbb{R}^{*\times7}`
          - :obj:`sim3_type`
        * - (:obj:`RxSO3_type`, :obj:`rxSO3_type`)
          - :math:`(\mathcal{G}\in\mathbb{R}^{*\times5}, \mathcal{g}\in\mathbb{R}^{*\times4})`
          - :math:`\mapsto`
          - :math:`\mathcal{g}\in\mathbb{R}^{*\times4}`
          - :obj:`rxso3_type`

    * If input (:math:`\mathbf{x}`, :math:`\mathbf{a}`)'s :obj:`ltype` are :obj:`SO3_type` and :obj:`so3_type`
      (input :math:`\mathbf{a}` is an instance of :meth:`so3`). Let :math:`\mathbf{y}` be the output.

        .. math::
            \mathbf{y}_i = \mathbf{J}^{-1}_i(\mathbf{x}_i)\mathbf{a}_i,

        where :math:`\mathbf{J}^{-1}_i(\mathbf{x}_i)` is the left-Jacobian of :math:`\mathbf{x}_i`. Let :math:`\boldsymbol{\phi}_i = \theta_i\mathbf{n}_i` 
        be the corresponding Lie Algebra of :math:`\mathbf{x}_i`, :math:`\boldsymbol{\Phi}_i` be the skew matrix of :math:`\boldsymbol{\phi}_i`:

        .. math::
            \mathbf{J}^{-1}_i(\mathbf{x}_i) = \mathbf{I} - \frac{1}{2}\boldsymbol{\Phi}_i + \mathrm{coef}\boldsymbol{\Phi}_i^2

        where :math:`\mathbf{I}` is the identity matrix with the same dimension as :math:`\boldsymbol{\Phi}_i`, and 

        .. math::
            \mathrm{coef} = \left\{
                                \begin{array}{ll} 
                                    \frac{1}{\theta^2} - \frac{\cos{\frac{\theta}{2}}}{2\theta\sin{\frac{\theta}{2}}}, \quad \|\theta\| > \text{eps}, \\
                                    \frac{1}{12},
                                    \quad \|\theta\| \leq \text{eps},
                                \end{array}
                             \right.

    Note:
        :math:`\mathrm{Jinvp}` is usually used in the Baker-Campbell-Hausdorff formula (BCH formula) when performing LieTensor multiplication.
        One can refer to this paper for more details:

        * J. Sola et al., `A micro Lie theory for state estimation in robotics <https://arxiv.org/abs/1812.01537>`_,
          arXiv preprint arXiv:1812.01537 (2018).
        
        In particular, Eq.(146) is the math used in the :obj:`SO3_type`, :obj:`so3_type` scenario. 
    
    Example:

        * :math:`\mathrm{Jinvp}`: (:obj:`SO3`, :obj:`so3`) :math:`\mapsto` :obj:`so3`

        >>> x = pp.randn_SO3(2)
        >>> p = pp.randn_so3(2)
        >>> x.Jinvp(p) # equivalent to: pp.Jinvp(x, a)
        so3Type LieTensor:
        tensor([[ 0.8782,  0.5898, -1.9071],
                [-0.6499, -0.3977,  0.8115]])


    """
    return input.Jinvp(p)


@assert_ltype
def Jr(x):
    return x.Jr()
