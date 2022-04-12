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
    LieTensor that satisfies the corresponding :code:`input.ltype`.

    The corresponding random generator can be

    .. list-table:: List of available random LieTensor generator of input :code:`ltype`.
        :widths: 25 25 30 30 30
        :header-rows: 1

        * - Name
          - ltype
          - randn function
          - Manifold
          - randn function
        * - Orthogonal Group
          - :code:`SO3_type`
          - :meth:`randn_SO3`
          - :code:`so3_type`
          - :meth:`randn_so3`
        * - Euclidean Group
          - :code:`SE3_type`
          - :meth:`randn_SE3`
          - :code:`se3_type`
          - :meth:`randn_se3`
        * - Similarity Group
          - :code:`Sim3_type`
          - :meth:`randn_Sim3`
          - :code:`sim3_type`
          - :meth:`randn_sim3`
        * - Scaling Orthogonal
          - :code:`RxSO3_type`
          - :meth:`randn_RxSO3`
          - :code:`rxso3_type`
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
    Returns :code:`so3_type` LieTensor filled with random numbers from a normal
    distribution with mean 0 and variance :code:`sigma` (also called the standard normal distribution).

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
        LieTensor: a :code:`so3_type` LieTensor

    Example:
        >>> pp.randn_so3(2, sigma=0.1, requires_grad=True, dtype=torch.float64)
        so3Type LieTensor:
        tensor([[-0.0427, -0.0149,  0.0948],
                [ 0.0607,  0.0473,  0.0703]], dtype=torch.float64, requires_grad=True)
    '''
    return so3_type.randn(*size, sigma=sigma, **kwargs)


def randn_SO3(*size, sigma=1, **kwargs):
    r'''
    Returns :code:`SO3_type` LieTensor filled with the Exponential map of the random
    :code:`so3_type` LieTensor with normal distribution with mean 0 and variance :code:`sigma`.

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
        LieTensor: a :code:`SO3_type` LieTensor

    Example:
        >>> pp.randn_SO3(2, sigma=0.1, requires_grad=True, dtype=torch.float64)
        SO3Type LieTensor:
        tensor([[-0.0060, -0.0517, -0.0070,  0.9986],
                [ 0.0015,  0.0753,  0.0503,  0.9959]], dtype=torch.float64, requires_grad=True)

    '''
    return SO3_type.randn(*size, sigma=sigma, **kwargs)


def randn_se3(*size, sigma=1, **kwargs):
    r'''
    Returns :code:`se3_type` LieTensor filled with random numbers from a normal
    distribution with mean 0 and variance :code:`sigma` (also called the standard normal distribution).

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
        LieTensor: a :code:`se3_type` LieTensor

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
    Returns :code:`SE3_type` LieTensor filled with the Exponential map of the random
    :code:`se3_type` LieTensor with normal distribution with mean 0 and variance :code:`sigma`.

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
        LieTensor: a :code:`SE3_type` LieTensor

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
    r"""The Exponential map for :code:`LieTensor` (Lie Algebra).

    .. math::
        \mathrm{Exp}: \mathcal{g} \mapsto \mathcal{G}

    Args:
        input (LieTensor): the input LieTensor (Lie Algebra)

    Return:
        LieTensor: the output LieTensor (Lie Group)

    .. list-table:: List of supported :math:`\mathrm{Exp}` map
        :widths: 20 20 8 20 20
        :header-rows: 1

        * - input :code:`ltype`
          - :math:`\mathcal{g}` (Lie Algebra)
          - :math:`\mapsto`
          - :math:`\mathcal{G}` (Lie Group)
          - output :code:`ltype`
        * - :code:`so3_type`
          - :math:`\mathcal{g}\in\mathbb{R}^{*\times3}`
          - :math:`\mapsto`
          - :math:`\mathcal{G}\in\mathbb{R}^{*\times4}`
          - :code:`SO3_type`
        * - :code:`se3_type`
          - :math:`\mathcal{g}\in\mathbb{R}^{*\times6}`
          - :math:`\mapsto`
          - :math:`\mathcal{G}\in\mathbb{R}^{*\times7}`
          - :code:`SE3_type`
        * - :code:`sim3_type`
          - :math:`\mathcal{g}\in\mathbb{R}^{*\times7}`
          - :math:`\mapsto`
          - :math:`\mathcal{G}\in\mathbb{R}^{*\times8}`
          - :code:`Sim3_type`
        * - :code:`rxso3_type`
          - :math:`\mathcal{g}\in\mathbb{R}^{*\times4}`
          - :math:`\mapsto`
          - :math:`\mathcal{G}\in\mathbb{R}^{*\times5}`
          - :code:`RxSO3_type`

    * Input :math:`\mathbf{x}`'s :code:`ltype` is :code:`so3_type` (input :math:`\mathbf{x}` is an instance of :meth:`so3`):

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

    Note:
        This function :func:`Exp()` is different from :func:`exp()`, which returns
        a new torch tensor with the exponential of the elements of the input tensor.

    Example:
        >>> x = pp.randn_so3(2, requires_grad=True)
        so3Type LieTensor:
        tensor([[ 0.1366,  0.1370, -1.1921],
                [-0.6003, -0.2165, -1.6576]], requires_grad=True)

        :meth:`Exp` returns LieTensor :meth:`SO3`.

        >>> x.Exp() # equivalent to: pp.Exp(x)
        SO3Type LieTensor:
        tensor([[ 0.0642,  0.0644, -0.5605,  0.8232],
                [-0.2622, -0.0946, -0.7241,  0.6309]], grad_fn=<AliasBackward0>)

        :meth:`exp` returns torch tensor.

        >>> x.exp() # Note that this returns torch tensor
        tensor([[1.1463, 1.1469, 0.3036],
                [0.5486, 0.8053, 0.1906]], grad_fn=<ExpBackward0>)
    """
    return input.Exp()


@assert_ltype
def Log(input):
    r"""The Logarithm map for :code:`LieTensor` (Lie Group).

    .. math::
        \mathrm{Log}: \mathcal{G} \mapsto \mathcal{g}

    Args:
        input (LieTensor): the input LieTensor (Lie Group)

    Return:
        LieTensor: the output LieTensor (Lie Algebra)

    .. list-table:: List of supported :math:`\mathrm{Log}` map
        :widths: 20 20 8 20 20
        :header-rows: 1

        * - input :code:`ltype`
          - :math:`\mathcal{G}` (Lie Group)
          - :math:`\mapsto`
          - :math:`\mathcal{g}` (Lie Algebra)
          - output :code:`ltype`
        * - :code:`SO3_type`
          - :math:`\mathcal{g}\in\mathbb{R}^{*\times4}`
          - :math:`\mapsto`
          - :math:`\mathcal{G}\in\mathbb{R}^{*\times3}`
          - :code:`so3_type`
        * - :code:`SE3_type`
          - :math:`\mathcal{g}\in\mathbb{R}^{*\times7}`
          - :math:`\mapsto`
          - :math:`\mathcal{G}\in\mathbb{R}^{*\times6}`
          - :code:`se3_type`
        * - :code:`Sim3_type`
          - :math:`\mathcal{g}\in\mathbb{R}^{*\times8}`
          - :math:`\mapsto`
          - :math:`\mathcal{G}\in\mathbb{R}^{*\times7}`
          - :code:`sim3_type`
        * - :code:`RxSO3_type`
          - :math:`\mathcal{g}\in\mathbb{R}^{*\times5}`
          - :math:`\mapsto`
          - :math:`\mathcal{G}\in\mathbb{R}^{*\times4}`
          - :code:`rxso3_type`
    
    Note:
        This function :func:`Log()` is different from :func:`log()`, which returns
        a new torch tensor with the logarithm of the elements of the input tensor.

    * If input :math:`\mathbf{x}`'s :code:`ltype` is :code:`SO3_type`
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

    * If input :math:`\mathbf{x}`'s :code:`ltype` is :code:`SE3_type`
      (input :math:`\mathbf{x}` is an instance of :meth:`SE3`):

        Let :math:`\mathbf{t}_i`, :math:`\mathbf{q}_i` be the translation and rotation parts of
        :math:`\mathbf{x}_i`, respectively; :math:`\mathbf{y}` be the output.

        .. math::
            \mathbf{y}_i = \left[\mathbf{J}_i^{-1}\mathbf{t}_i, \mathrm{Log}(\mathbf{q}_i) \right],

        where :math:`\mathrm{Log}` is the Logarithm map for :code:`SO3_type` input and
        :math:`\mathbf{J}_i` is the Jacobian matrix.

    * If input :math:`\mathbf{x}`'s :code:`ltype` is :code:`RxSO3_type`
      (input :math:`\mathbf{x}` is an instance of :meth:`RxSO3`):

        Let :math:`\mathbf{q}_i`, :math:`s_i` be the rotation and scale parts of :math:`\mathbf{x}_i`, respectively;
        :math:`\mathbf{y}` be the output.

        .. math::
            \mathbf{y}_i = \left[\mathrm{Log}(\mathbf{q}_i), \log(s_i) \right].

    * If input :math:`\mathbf{x}`'s :code:`ltype` is :code:`Sim3_type` (input :math:`\mathbf{x}`
      is an instance of :meth:`Sim3`):

        Let :math:`\mathbf{t}_i`, :math:`^s\mathbf{q}_i` be the translation and :code:`RxSO3` parts
        of :math:`\mathbf{x}_i`, respectively; :math:`\mathbf{y}` be the output.

        .. math::
            \mathbf{y}_i = \left[^s\mathbf{J}_i^{-1}\mathbf{t}_i, \mathrm{Log}(^s\mathbf{q}_i) \right],

        where :math:`^s\mathbf{J}` is the similarity transformed Jacobian matrix.

    Note:
        The :math:`\mathrm{arctan}`-based Logarithm map implementation thanks to the paper:

        * C. Hertzberg et al., `Integrating Generic Sensor Fusion Algorithms with Sound State
          Representation through Encapsulation of Manifolds <https://doi.org/10.1016/j.inffus.2011.08.003>`_,
          Information Fusion, 2013.

        Assume we have a unit rotation axis :math:`\mathbf{n}` and rotation angle :math:`\theta`, then
        its quaternion :math:`\mathbf{q}=[\boldsymbol{\nu}, w]`, where :math:`w` is
        the scalar part, can be defined as

            .. math::
                \mathbf{q} = \left[\sin(\theta/2) \mathbf{n}, \cos(\theta/2) \right]

        Therefore, given a quaternion, we can calculate its rotation angle :math:`\theta` as

            .. math::
                \theta = 2\mathrm{arctan}(\|\boldsymbol{\nu}\|/w),~\|\boldsymbol{\nu}\| = \sin(\theta/2).

    Example:

        * :math:`\mathrm{Log}`: :code:`SO3` :math:`\mapsto` :code:`so3`

        >>> x = pp.randn_SO3(2)
        >>> x.Log() # equivalent to: pp.Log(x)
        so3Type LieTensor:
        tensor([[-0.3060,  0.2344,  1.2724],
                [ 0.3012, -0.6817,  0.1187]])

        * :math:`\mathrm{Log}`: :code:`SE3` :math:`\mapsto` :code:`se3`

        >>> x = pp.randn_SE3(2)
        >>> x.Log() # equivalent to: pp.Log(x)
        se3Type LieTensor:
        tensor([[ 0.2958, -0.0840, -1.4733,  0.7004,  0.4483, -0.9009],
                [ 0.0850, -0.1020, -1.2616, -1.0524, -1.2031,  0.8377]])


        * :math:`\mathrm{Log}`: :code:`RxSO3` :math:`\mapsto` :code:`rxso3`

        >>> x = pp.randn_RxSO3(2)
        >>> x.Log() # equivalent to: pp.Log(x)
        rxso3Type LieTensor:
        tensor([[-1.3755,  0.3525, -2.2367,  0.5409],
                [ 0.5929, -0.3250, -0.7394,  1.0965]])

        * :math:`\mathrm{Log}`: :code:`Sim3` :math:`\mapsto` :code:`sim3`

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
def Jinv(X, a):
    return X.Jinv(a)


@assert_ltype
def Jr(x):
    return x.Jr()
