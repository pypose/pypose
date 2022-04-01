import torch
import functools
from .groups import  LieGroup
from .groups import SE3_type, se3_type
from .groups import SO3_type, so3_type
from .groups import Sim3_type, sim3_type
from .groups import RxSO3_type, rxso3_type


SO3 = functools.partial(LieGroup, gtype=SO3_type)
so3 = functools.partial(LieGroup, gtype=so3_type)
SE3 = functools.partial(LieGroup, gtype=SE3_type)
se3 = functools.partial(LieGroup, gtype=se3_type)
Sim3 = functools.partial(LieGroup, gtype=Sim3_type)
sim3 = functools.partial(LieGroup, gtype=sim3_type)
RxSO3 = functools.partial(LieGroup, gtype=RxSO3_type)
rxso3 = functools.partial(LieGroup, gtype=rxso3_type)


def randn_like(input, sigma=1, **kwargs):
    return input.gtype.randn_like(*input.gshape, sigma=sigma, **kwargs)


def randn_so3(*size, sigma=1, **kwargs):
    r'''
    Returns :code:`so3_type` LieGroup tensor filled with random numbers from a normal
    distribution with mean 0 and variance :code:`sigma` (also called the standard normal distribution).

    .. math::
        \mathrm{out}_i = \mathcal{N}(\mathbf{0}_{3\times 1}, \mathbf{\sigma}_{3\times 1})

    The shape of the tensor is defined by the variable argument size.

    Args:
        size (int...): a sequence of integers defining the shape of the output tensor.
            Can be a variable number of arguments or a collection like a list or tuple.

        sigma (float): variance of the normal distribution. Default: 1.

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
        LieGroup: a :code:`so3_type` LieGroup Tensor

    Example:
        >>> pp.randn_so3(2, sigma=0.1, requires_grad=True, dtype=torch.float64)
        so3Type Group:
        tensor([[-0.0427, -0.0149,  0.0948],
                [ 0.0607,  0.0473,  0.0703]], dtype=torch.float64, requires_grad=True)
    '''
    return so3_type.randn(*size, sigma=sigma, **kwargs)


def randn_SO3(*size, sigma=1, **kwargs):
    r'''
    Returns :code:`SO3_type` LieGroup tensor filled with the Exponential map of the random
    :code:`so3_type` LieGroup tensor with normal distribution with mean 0 and variance :code:`sigma`.

    .. math::
        \mathrm{out}_i = \mathrm{Exp}(\mathcal{N}(\mathbf{0}_{3\times 1}, \mathbf{\sigma}_{3\times 1}))

    The shape of the tensor is defined by the variable argument size.

    Args:
        size (int...): a sequence of integers defining the shape of the output tensor.
            Can be a variable number of arguments or a collection like a list or tuple.

        sigma (float): variance :math:`\sigma` of the normal distribution. Default: 1.

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
        LieGroup: a :code:`SO3_type` LieGroup Tensor

    Example:
        >>> pp.randn_SO3(2, sigma=0.1, requires_grad=True, dtype=torch.float64)
        SO3Type Group:
        tensor([[-0.0060, -0.0517, -0.0070,  0.9986],
                [ 0.0015,  0.0753,  0.0503,  0.9959]], dtype=torch.float64, requires_grad=True)

    '''
    return SO3_type.randn(*size, sigma=sigma, **kwargs)


def randn_se3(*args, sigma=1, requires_grad=False, **kwargs):
    return se3_type.randn(*args, sigma=sigma, requires_grad=requires_grad, **kwargs)


def randn_SE3(*args, sigma=1, requires_grad=False, **kwargs):
    return SE3_type.randn(*args, sigma=sigma, requires_grad=requires_grad, **kwargs)


def randn_sim3(*args, sigma=1, requires_grad=False, **kwargs):
    return sim3_type.randn(*args, sigma=sigma, requires_grad=requires_grad, **kwargs)


def randn_Sim3(*args, sigma=1, requires_grad=False, **kwargs):
    return Sim3_type.randn(*args, sigma=sigma, requires_grad=requires_grad, **kwargs)


def randn_rxso3(*args, sigma=1, requires_grad=False, **kwargs):
    return rxso3_type.randn(*args, sigma=sigma, requires_grad=requires_grad, **kwargs)


def randn_RxSO3(*args, sigma=1, requires_grad=False, **kwargs):
    return RxSO3_type.randn(*args, sigma=sigma, requires_grad=requires_grad, **kwargs)


def identity_like(liegroup, **kwargs):
    return liegroup.gtype.identity_like(*liegroup.gshape, **kwargs)


def identity_SO3(*args, **kwargs):
    return SO3_type.identity(*args, **kwargs)


def identity_so3(*args, **kwargs):
    return so3_type.identity(*args, **kwargs)


def identity_SE3(*args, **kwargs):
    return SE3_type.identity(*args, **kwargs)


def identity_se3(*args, **kwargs):
    return se3_type.identity(*args, **kwargs)


def identity_sim3(*args, **kwargs):
    return sim3_type.identity(*args, **kwargs)


def identity_Sim3(*args, **kwargs):
    return Sim3_type.identity(*args, **kwargs)


def identity_rxso3(*args, **kwargs):
    return rxso3_type.identity(*args, **kwargs)


def identity_RxSO3(*args, **kwargs):
    return RxSO3_type.identity(*args, **kwargs)


def assert_gtype(func):
    @functools.wraps(func)
    def checker(*args, **kwargs):
        assert isinstance(args[0], LieGroup), "Invalid LieGroup Type."
        out = func(*args, **kwargs)
        return out
    return checker


@assert_gtype
def Exp(input):
    r"""The Exponential map for :code:`LieGroup` Tensor.

    .. math::
        \exp: \mathcal{g} \mapsto \mathcal{G}

    .. list-table:: List of supported :math:`\exp` map
        :widths: 30 30 30 30
        :header-rows: 1

        * - input :code:`gtype`
          - :math:`\mathcal{g}`
          - :math:`\mathcal{G}`
          - output :code:`gtype`
        * - :code:`so3_type`
          - :math:`\mathcal{g}\in\mathbb{R}^3`
          - :math:`\mathcal{G}\in\mathbb{R}^4`
          - :code:`SO3_type`
        * - :code:`se3_type`
          - :math:`\mathcal{g}\in\mathbb{R}^6`
          - :math:`\mathcal{G}\in\mathbb{R}^7`
          - :code:`SE3_type`
        * - :code:`sim3_type`
          - :math:`\mathcal{g}\in\mathbb{R}^7`
          - :math:`\mathcal{G}\in\mathbb{R}^8`
          - :code:`Sim3_type`
        * - :code:`rxso3_type`
          - :math:`\mathcal{g}\in\mathbb{R}^4`
          - :math:`\mathcal{G}\in\mathbb{R}^5`
          - :code:`RxSO3_type`

    Args:
        input (LieGroup): the input LieGroup Tensor on manifold

    Return:
        LieGroup: The LieGroup Tensor in embedding space

    Note:
        This function :func:`Exp()` is different from :func:`exp()`, which returns
        a new torch tensor with the exponential of the elements of the input tensor.

    Example:
        >>> x = pp.randn_so3(2, requires_grad=True)
        so3Type Group:
        tensor([[ 0.1366,  0.1370, -1.1921],
                [-0.6003, -0.2165, -1.6576]], requires_grad=True)

        >>> x.Exp() # equivalent to: pp.Exp(x)
        SO3Type Group:
        tensor([[ 0.0642,  0.0644, -0.5605,  0.8232],
                [-0.2622, -0.0946, -0.7241,  0.6309]], grad_fn=<AliasBackward0>)

        >>> x.exp() # Note that this returns torch.Tensor
        tensor([[1.1463, 1.1469, 0.3036],
                [0.5486, 0.8053, 0.1906]], grad_fn=<ExpBackward0>)
    """
    return input.Exp()


@assert_gtype
def Log(input):
    r"""The Logarithm map for :code:`LieGroup` Tensor.

    .. math::
        \log: \mathcal{G} \mapsto \mathcal{g}

    .. list-table:: List of supported :math:`\log` map
        :widths: 30 30 30 30
        :header-rows: 1

        * - input :code:`gtype`
          - :math:`\mathcal{G}`
          - :math:`\mathcal{g}`
          - output :code:`gtype`
        * - :code:`SO3_type`
          - :math:`\mathcal{g}\in\mathbb{R}^4`
          - :math:`\mathcal{G}\in\mathbb{R}^3`
          - :code:`so3_type`
        * - :code:`SE3_type`
          - :math:`\mathcal{g}\in\mathbb{R}^7`
          - :math:`\mathcal{G}\in\mathbb{R}^6`
          - :code:`se3_type`
        * - :code:`Sim3_type`
          - :math:`\mathcal{g}\in\mathbb{R}^8`
          - :math:`\mathcal{G}\in\mathbb{R}^7`
          - :code:`sim3_type`
        * - :code:`RxSO3_type`
          - :math:`\mathcal{g}\in\mathbb{R}^5`
          - :math:`\mathcal{G}\in\mathbb{R}^4`
          - :code:`rxso3_type`

    Args:
        input (LieGroup): the input LieGroup Tensor on manifold

    Return:
        LieGroup: The LieGroup Tensor in embedding space

    Note:
        This function :func:`Log()` is different from :func:`log()`, which returns
        a new torch tensor with the logarithm of the elements of the input tensor.

    Example:
        >>> x = pp.randn_SO3(2, requires_grad=True)
        SO3Type Group:
        tensor([[-0.1420,  0.1088,  0.5904,  0.7871],
                [ 0.1470, -0.3328,  0.0580,  0.9297]], requires_grad=True)

        >>> x.Log() # equivalent to: pp.Log(x)
        so3Type Group:
        tensor([[-0.3060,  0.2344,  1.2724],
                [ 0.3012, -0.6817,  0.1187]], grad_fn=<AliasBackward0>)

        >>> x.log() # Note that this returns torch.Tensor
        tensor([[    nan, -2.2184, -0.5270, -0.2395],
                [-1.9171,     nan, -2.8478, -0.0729]], grad_fn=<LogBackward0>)
    """
    return input.Log()


@assert_gtype
def Inv(x):
    return x.Inv()


@assert_gtype
def Mul(x, y):
    return x * y


@assert_gtype
def Retr(X, a):
    return X.Retr(a)


@assert_gtype
def Act(X, p):
    return X.Act(p)


@assert_gtype
def Adj(X, a):
    return X.Adj(a)


@assert_gtype
def AdjT(X, a):
    return X.AdjT(a)


@assert_gtype
def Jinv(X, a):
    return X.Jinv(a)
