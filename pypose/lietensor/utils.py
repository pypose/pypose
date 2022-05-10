import torch
import functools
from .lietensor import  LieTensor
from .lietensor import LieType
from .lietensor import SE3_type, se3_type
from .lietensor import SO3_type, so3_type
from .lietensor import Sim3_type, sim3_type
from .lietensor import RxSO3_type, rxso3_type


def _LieTensor_wrapper_add_docstr(wrapper: functools.partial, embedding_doc):
    ltype: LieType = wrapper.keywords['ltype']
    type_name = type(ltype).__name__.removesuffix('Type')
    type_dim = ltype.dimension[0]
    see_method = ['Exp', 'Inv'] if ltype.on_manifold else \
        ['Log', 'Inv', 'Act', 'Retr', 'Adj', 'AdjT', 'Jinvp']
    wrapper.__doc__ = fr'''Creates a {type_name} :obj:`LieTensor`.

    Args:
        data (:obj:`Tensor`, or :obj:`list`, or ':obj:`int`...'): A
            :obj:`Tensor` object, or constructing a :obj:`Tensor`
            object from :obj:`list`, which defines tensor data (see below), or
            from ':obj:`int`...', which defines tensor shape.

            The shape of :obj:`Tensor` object must be `(*, {type_dim})`,
            where `*` is empty, one, or more batched dimensions (the
            :obj:`~LieTensor.lshape` of this LieTensor), otherwise error will
            be raised.

    {embedding_doc}

    If `data` is tensor-like, the last dimension should correspond to the
    {type_dim} elements of the above embedding.

    Note:
        It is not advised to construct {type_name} Tensors by specifying storage
        sizes with `int...`, which does not initialize data.

        Consider using :obj:`pypose.randn_{type_name}` or
        :obj:`pypose.identity_{type_name}` instead.

    See {', '.join([f':obj:`pypose.{m}`' for m in see_method])} for
    implementations of relavant operations.
    '''
    return wrapper

SO3 = _LieTensor_wrapper_add_docstr(functools.partial(LieTensor, ltype=SO3_type),
    r'''Internally, SO3 LieTensors are stored as unit quaternions:

    .. math::
        \mathrm{data}[*, :] = [q_x, q_y, q_z, q_w]

    where :math:`q_x^2 + q_y^2 + q_z^2 + q_w^2 = 1`.
    
    Note:
        Normalization is not required at initialization as it is done internally
        by the library right before further computation. However, the normalized
        quaternion will not be written back to the tensor storage to prevent
        in-place data alteration.

    Examples:
        >>> pp.SO3(torch.randn(2, 4))
        SO3Type LieTensor:
        tensor([[-1.0722, -0.9440,  0.9437, -0.8485],
                [-0.2725,  0.8414, -1.0730,  1.3270]])
        >>> pp.SO3([0, 0, 0, 1])
        SO3Type LieTensor:
        tensor([0., 0., 0., 1.])
    ''')

so3 = _LieTensor_wrapper_add_docstr(functools.partial(LieTensor, ltype=so3_type),
    r'''Internally, so3 LieTensors are stored in the
    `axis-angle <https://en.wikipedia.org/wiki/Axis-angle_representation>`_ format:

    .. math::
        \mathrm{data}[*, :] = [\delta_x, \delta_y, \delta_z]

    with :math:`\delta = \begin{pmatrix} \delta_x & \delta_y & \delta_z \end{pmatrix}^T`
    being the axis of rotation and :math:`\theta = \|{\delta}\|` being the angle.

    Examples:
        >>> pp.so3(torch.randn(2, 3))
        so3Type LieTensor:
        tensor([[ 0.1571,  0.2203, -0.2457],
                [-0.3311,  0.5412, -0.7028]])
        >>> pp.so3([0, 0, 1])
        so3Type LieTensor:
        tensor([0., 0., 1.])
    ''')

SE3 = _LieTensor_wrapper_add_docstr(functools.partial(LieTensor, ltype=SE3_type),
    r'''Internally, SE3 LieTensors are stored by concatenating the unit quaternion
    representing the rotation with a vector representing the translation.

    .. math::
        \mathrm{data}[*, :] = [t_x, t_y, t_z, q_x, q_y, q_z, q_w]

    where :math:`\begin{pmatrix} t_x & t_y & t_z \end{pmatrix}^T \in \mathbb{R}^3` is
    the translation and
    :math:`\begin{pmatrix} q_x & q_y & q_z & q_w \end{pmatrix}^T` is the unit
    quaternion as in :obj:`pypose.SO3`.

    Examples:
        >>> pp.SE3(torch.randn(2, 7))
        SE3Type LieTensor:
        tensor([[ 0.1626,  1.6349,  0.3607,  0.2848, -0.0948,  0.1541,  1.0003],
                [ 1.4034, -1.3085, -0.8886, -1.6703,  0.7381,  1.5575,  0.6280]])
        >>> pp.SE3([0, 0, 0, 0, 0, 0, 1])
        SE3Type LieTensor:
        tensor([0., 0., 0., 0., 0., 0., 1.])
    ''')

se3 = _LieTensor_wrapper_add_docstr(functools.partial(LieTensor, ltype=se3_type),
    r'''Internally, se3 LieTensors are stored by concatenating the axis-angle
    representation of the rotation with the "velocity" vector:

    .. math::
        \mathrm{data}[*, :] = [\delta_x, \delta_y, \delta_z, \log t_x, \log t_y, \log t_z]

    where :math:`\begin{pmatrix} \delta_x & \delta_y & \delta_z \end{pmatrix}^T`
    is the axis-angle vector as in :obj:`pypose.so3`, and
    :math:`\begin{pmatrix} \log t_x & \log t_y & \log t_z \end{pmatrix}^T`
    is the element-wise logarithm of the translation vector from :obj:`pypose.SE3`.

    Examples:
        >>> pp.se3(torch.randn(2, 6))
        se3Type LieTensor:
        tensor([[-0.8710, -1.4994, -0.2843,  1.0185, -0.3932, -0.4839],
                [-0.4750, -0.4804, -0.7083, -1.8141, -1.4409, -0.3125]])
        >>> pp.se3([0, 0, 0, 0, 0, 1])
        se3Type LieTensor:
        tensor([0., 0., 0., 0., 0., 1.])
    ''')

RxSO3 = _LieTensor_wrapper_add_docstr(functools.partial(LieTensor, ltype=RxSO3_type),
    r'''Internally, RxSO3 LieTensors are stored by concatenating the unit quaternion
    representing the rotation with a scaling factor:

    .. math::
        \mathrm{data}[*, :] = [q_x, q_y, q_z, q_w, s]

    where :math:`\begin{pmatrix} q_x & q_y & q_z & q_w \end{pmatrix}^T`
    is the unit quaternion as in :obj:`pypose.SO3` and
    :math:`s \in \mathbb{R}` is the scaling factor.

    Examples:
        >>> pp.RxSO3(torch.randn(2, 5))
        RxSO3Type LieTensor:
        tensor([[-0.3693,  2.5155, -0.5384, -0.8119, -0.4798],
                [-0.4058, -0.5909, -0.4918, -0.2994,  0.5440]])
        >>> pp.RxSO3([0, 0, 0, 1, 1])
        RxSO3Type LieTensor:
        tensor([0., 0., 0., 1., 1.])
    ''')

rxso3 = _LieTensor_wrapper_add_docstr(functools.partial(LieTensor, ltype=rxso3_type),
    r'''Internally, rxso3 LieTensors are stored by concatenating the axis-angle
    representation of the rotation with the log scale:

    .. math::
        \mathrm{data}[*, :] = [\delta_x, \delta_y, \delta_z, \log s]

    where :math:`\begin{pmatrix} \delta_x & \delta_y & \delta_z \end{pmatrix}^T`
    is the axis-angle vector in :obj:`pypose.so3`, and
    :math:`s \in \mathbb{R}` is the scaling factor in :obj:`pypose.RxSO3`.

    Examples:
        >>> pp.rxso3(torch.randn(2, 4))
        rxso3Type LieTensor:
        tensor([[ 0.3752, -0.1576,  1.2057,  0.6086],
                [ 0.8434,  0.2449,  0.0488, -0.1202]])
        >>> pp.rxso3([0, 0, 0, 0, 1])
        RxSO3Type LieTensor:
        tensor([0., 0., 0., 0., 1.])
    ''')

Sim3 = _LieTensor_wrapper_add_docstr(functools.partial(LieTensor, ltype=Sim3_type),
    r'''Internally, Sim3 LieTensors are stored by concatenating the translation
    vector with an RxSO3:

    .. math::
        \mathrm{data}[*, :] = [t_x, t_y, t_z, q_x, q_y, q_z, q_w, s]

    where :math:`\begin{pmatrix} t_x & t_y & t_z \end{pmatrix}^T \in \mathbb{R}^3`
    is the translation vector and
    :math:`\begin{pmatrix} q_x & q_y & q_z & q_w \end{pmatrix}^T` and
    :math:`s \in \mathbb{R}` are the unit quaternion and the scaling factor
    as in :obj`pp.RxSO3`, respectively.

    Examples:
        >>> pp.Sim3(torch.randn(2, 8))
        Sim3Type LieTensor:
        tensor([[ 0.0175,  0.8657, -0.2274,  2.2380, -0.0297, -0.3799, -0.0664,  0.9995],
                [ 0.8744,  0.4114,  1.2041, -0.5687, -0.5630,  0.6025, -0.6137,  1.1185]])
        >>> pp.Sim3([0, 0, 0, 0, 0, 0, 1, 1])
        Sim3Type LieTensor:
        tensor([0., 0., 0., 0., 0., 0., 1., 1.])
    ''')

sim3 = _LieTensor_wrapper_add_docstr(functools.partial(LieTensor, ltype=sim3_type),
    r'''Internally, sim3 LieTensors are stored by concatenating the log translation
    vector with the corresponding rxso3:

    .. math::
        \mathrm{data}[*, :] = [\log t_x, \log t_y, \log t_z, \delta_x, \delta_y, \delta_z, \log s]

    where :math:`\begin{pmatrix} \log t_x & \log t_y & \log t_z \end{pmatrix}^T`
    is the log translation vector as in :obj:`pypose.se3`, and
    :math:`\begin{pmatrix} \delta_x & \delta_y & \delta_z & \log s \end{pmatrix}^T`
    represents the rotation and scaling, as in :obj:`pypose.rxso3`.

    Examples:
        >>> pp.Sim3(torch.randn(2, 7))
        sim3Type LieTensor:
        sim3Type LieTensor:
        tensor([[ 0.1477, -1.3500, -2.1571,  0.8893, -0.7821, -0.9889, -0.7887],
                [ 0.2251,  0.3512,  0.0485,  0.0163, -1.7090, -0.0417, -0.3842]])
        >>> pp.sim3([0, 0, 0, 0, 0, 0, 1])
        sim3Type LieTensor:
        tensor([0., 0., 0., 0., 0., 0., 1.])
    ''')

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
    Returns :obj:`so3_type` LieTensor filled with random numbers satisfying the expected distance 
    between the corresponding state and :math:`\mathbf{0}_{3\times 1}` is :obj:`sigma`.

    The shape of the tensor is defined by the variable argument size.

    Args:
        size (int...): a sequence of integers defining the shape of the output tensor.
            Can be a variable number of arguments or a collection like a list or tuple.

        sigma (float, optional): expected distance between the generated state and zero. Default: 1.

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

    Given the expected distance :obj:`sigma`, we first calculte the standard deviation of the 
    individual components of the tangent perturbation :math:`\sigma_{\mathrm{r}}` as:

    .. math::
        \sigma_{\mathrm{r}} = \frac{2*\sigma}{\sqrt{3}}.

    The factor 2 is due to the way we define distance (see also `Matt Mason's lecture on 
    quaternions <http://www.cs.cmu.edu/afs/cs/academic/class/16741-s07/www/lectures/Lecture8.pdf>`_)
    The :math:`1/\sqrt{3}` factor is necessary because the distribution in the tangent space is 
    a 3-dimensional Gaussian, so that the *length* of a tangent vector needs to be scaled by :math:`1/\sqrt{3}`.

    Then the output can be written as:

    .. math::
        \mathrm{out}_i = \mathcal{N}(\mathbf{0}_{3\times 1}, \mathbf{\sigma_{\mathrm{r}}}_{3\times 1}),

    where :math:`\mathcal{N}` denotes Gaussian distribution.

    Note:
        The detailed explanation of the above implementation can be found in the 
        `OMPL code <https://ompl.kavrakilab.org/SO3StateSpace_8cpp_source.html>`_, line 119.

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
    r'''
     Returns identity LieTensor with the same :obj:`lsize` and :obj:`ltype` as the given LieTensor.

    Args:
        liegroup (LieTensor): the size of liegroup will determine the size of the output tensor. 

    Args:
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
    
    Example:
        >>> x = pp.randn_SO3(3, device="cuda:0", dtype=torch.double, requires_grad=True)
        >>> pp.identity_like(x, device="cpu")
        SO3Type LieTensor:
        tensor([[0., 0., 0., 1.],
                [0., 0., 0., 1.],
                [0., 0., 0., 1.]])
    '''
    return liegroup.ltype.identity_like(*liegroup.lshape, **kwargs)


def identity_SO3(*lsize, **kwargs):
    r'''
    Returns identity :obj:`SO3_type` LieTensor with the given :obj:`lsize`.

    Args:
        lsize (int..., optional): a sequence of integers defining the :obj:`LieTensor.lshape` of
            the output LieTensor. Can be a variable number of arguments or a collection like a
            list or tuple. If not given, a single :obj:`SO3_type` item will be returned.

    Args:
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
        >>> pp.identity_SO3()
        SO3Type LieTensor:
        tensor([0., 0., 0., 1.])

        >>> pp.identity_SO3(2)
        SO3Type LieTensor:
        tensor([[0., 0., 0., 1.],
                [0., 0., 0., 1.]])

        >>> pp.identity_SO3(2, 1)
        SO3Type LieTensor:
        tensor([[[0., 0., 0., 1.]],
                [[0., 0., 0., 1.]]])
    '''
    return SO3_type.identity(*lsize, **kwargs)


def identity_so3(*lsize, **kwargs):
    r'''
    Returns identity :obj:`so3_type` LieTensor with the given :obj:`lsize`.

    Args:
        lsize (int..., optional): a sequence of integers defining the :obj:`LieTensor.lshape` of
            the output LieTensor. Can be a variable number of arguments or a collection like a
            list or tuple. If not given, a single :obj:`so3_type` item will be returned.

    Args:
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
        >>> pp.identity_so3()
        so3Type LieTensor:
        tensor([0., 0., 0.])

        >>> pp.identity_so3(2)
        so3Type LieTensor:
        tensor([[0., 0., 0.],
                [0., 0., 0.]])

        >>> pp.identity_so3(2,1)
        so3Type LieTensor:
        tensor([[[0., 0., 0.]],
                [[0., 0., 0.]]])
    '''
    return so3_type.identity(*lsize, **kwargs)


def identity_SE3(*lsize, **kwargs):
    r'''
    Returns identity :obj:`SE3_type` LieTensor with the given :obj:`lsize`.

    Args:
        lsize (int..., optional): a sequence of integers defining the :obj:`LieTensor.lshape` of
            the output LieTensor. Can be a variable number of arguments or a collection like a
            list or tuple. If not given, a single :obj:`SE3_type` item will be returned.
    
    Args:

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
        >>> pp.identity_SE3()
        SE3Type LieTensor:
        tensor([0., 0., 0., 0., 0., 0., 1.])

        >>> pp.identity_SE3(2)
        SE3Type LieTensor:
        tensor([[0., 0., 0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 0., 1.]])

        >>> pp.identity_SE3(2,1)
        SE3Type LieTensor:
        tensor([[[0., 0., 0., 0., 0., 0., 1.]],
                [[0., 0., 0., 0., 0., 0., 1.]]])
    '''
    return SE3_type.identity(*lsize, **kwargs)


def identity_se3(*lsize, **kwargs):
    r'''
    Returns identity :obj:`se3_type` LieTensor with the given :obj:`lsize`.
    
    Args:
        lsize (int..., optional): a sequence of integers defining the :obj:`LieTensor.lshape` of
            the output LieTensor. Can be a variable number of arguments or a collection like a
            list or tuple. If not given, a single :obj:`se3_type` item will be returned.

    Args:
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
        >>> pp.identity_se3()
        se3Type LieTensor:
        tensor([0., 0., 0., 0., 0., 0.])

        >>> pp.identity_se3(2)
        se3Type LieTensor:
        tensor([[0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.]])

        >>> pp.identity_se3(2,1)
        se3Type LieTensor:
        tensor([[0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.]])
    '''
    return se3_type.identity(*lsize, **kwargs)


def identity_sim3(*lsize, **kwargs):
    r'''
     Returns identity :obj:`sim3_type` LieTensor with the given :obj:`lsize`. 

    Args:
        lsize (int..., optional): a sequence of integers defining the :obj:`LieTensor.lshape` of
            the output LieTensor. Can be a variable number of arguments or a collection like a
            list or tuple. If not given, a single :obj:`sim3_type` item will be returned.

    Args:
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
        LieTensor: a :obj:`sim3_type` LieTensor
        
    Example:
        >>> pp.identity_sim3()
        sim3Type LieTensor:
        tensor([0., 0., 0., 0., 0., 0., 0.])

        >>> identity_sim3(2)
        sim3Type LieTensor:
        tensor([[0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0.]])

        >>> identity_sim3(2,1)
        sim3Type LieTensor:
        tensor([[[0., 0., 0., 0., 0., 0., 0.]],
                [[0., 0., 0., 0., 0., 0., 0.]]])
    '''
    return sim3_type.identity(*lsize, **kwargs)


def identity_Sim3(*lsize, **kwargs):
    r'''
    Returns identity :obj:`Sim3_type` LieTensor with the given :obj:`lsize`.

    Args:
        lsize (int..., optional): a sequence of integers defining the :obj:`LieTensor.lshape` of
            the output LieTensor. Can be a variable number of arguments or a collection like a
            list or tuple. If not given, a single :obj:`Sim3_type` item will be returned.

    Args:
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
        LieTensor: a :obj:`Sim3_type` LieTensor
        
    Example:
        >>> pp.identity_Sim3()
        Sim3Type LieTensor:
        tensor([0., 0., 0., 0., 0., 0., 1., 1.])

        >>> identity_Sim3(2)
        Sim3Type LieTensor:
        tensor([[0., 0., 0., 0., 0., 0., 1., 1.],
                [0., 0., 0., 0., 0., 0., 1., 1.]])

        >>> identity_Sim3(2,1)
        Sim3Type LieTensor:
        tensor([[[0., 0., 0., 0., 0., 0., 1., 1.]],
                [[0., 0., 0., 0., 0., 0., 1., 1.]]])
    '''
    return Sim3_type.identity(*lsize, **kwargs)    


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

    Warning:
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

        Let :math:`\bm{\tau}_i`, :math:`\bm{\phi}_i` be the translation and rotation parts
        of :math:`\mathbf{x}_i`, respectively; :math:`\mathbf{y}` be the output.

        .. math::
            \mathbf{y}_i = \left[\mathbf{J}_i\bm{\tau}_i, \mathrm{Exp}(\bm{\phi}_i)\right],
        
        where :math:`\mathrm{Exp}` is the Exponential map for :obj:`so3_type` input and
        :math:`\mathbf{J}_i` is the left Jacobian for :obj:`so3_type` input.

    * Input :math:`\mathbf{x}`'s :obj:`ltype` is :obj:`rxso3_type`
      (input :math:`\mathbf{x}` is an instance of :meth:`rxso3`):

        Let :math:`\bm{\phi}_i`, :math:`\sigma_i` be the rotation and scale parts of
        :math:`\mathbf{x}_i`, respectively; :math:`\mathbf{y}` be the output.

        .. math::
            \mathbf{y}_i = \left[\mathrm{Exp}(\bm{\phi}_i), \mathrm{exp}(\sigma_i)\right],

        where :math:`\mathrm{exp}` is the exponential function.

    * Input :math:`\mathbf{x}`'s :obj:`ltype` is :obj:`sim3_type`
      (input :math:`\mathbf{x}` is an instance of :meth:`sim3`):

        Let :math:`\bm{\tau}_i`, :math:`^{s}\bm{\phi}_i` be the translation and
        :meth:`rxso3` parts of :math:`\mathbf{x}_i`, respectively.
        :math:`\bm{\phi}_i = \theta_i\mathbf{n}_i`, :math:`\sigma_i` be the rotation
        and scale parts of :math:`^{s}\bm{\phi}_i`, :math:`\boldsymbol{\Phi}_i` be the skew matrix
        of :math:`\bm{\phi}_i`; :math:`s_i = e^\sigma_i`, :math:`\mathbf{y}` be the output.

        .. math::
            \mathbf{y}_i = \left[^{s}\mathbf{W}_i\bm{\tau}_i, \mathrm{Exp}(^{s}\bm{\phi}_i)\right],
        
        where

        .. math::
            ^s\mathbf{W}_i = A\boldsymbol{\Phi}_i + B\boldsymbol{\Phi}_i^2 + C\mathbf{I}

        in which if :math:`\|\sigma_i\| \geq \text{eps}`:

        .. math::
            A = \left\{
                    \begin{array}{ll} 
                        \frac{s_i\sin\theta_i\sigma_i + (1-s_i\cos\theta_i)\theta_i}
                        {\theta_i(\sigma_i^2 + \theta_i^2)}, \quad \|\theta_i\| \geq \text{eps}, \\
                        \frac{(\sigma_i-1)s_i+1}{\sigma_i^2}, \quad \|\theta_i\| < \text{eps},
                    \end{array}
                \right.

        .. math::
            B = 
            \left\{
                \begin{array}{ll} 
                    \left( C - \frac{(s_i\cos\theta_i-1)\sigma+ s_i\sin\theta_i\sigma_i}
                    {\theta_i^2+\sigma_i^2}\right)\frac{1}{\theta_i^2}, \quad \|\theta_i\| \geq \text{eps}, \\
                    \frac{s_i\sigma_i^2/2 + s_i-1-\sigma_i s_i}{\sigma_i^3}, \quad \|\theta_i\| < \text{eps},
                \end{array}
            \right.

        .. math::
            C = \frac{e^{\sigma_i} - 1}{\sigma_i}\mathbf{I}

        otherwise:

        .. math::
            A = \left\{
                    \begin{array}{ll} 
                        \frac{1-\cos\theta_i}{\theta_i^2}, \quad \|\theta_i\| \geq \text{eps}, \\
                        \frac{1}{2}, \quad \|\theta_i\| < \text{eps},
                    \end{array}
                \right.

        .. math::
            B = \left\{
                    \begin{array}{ll} 
                        \frac{\theta_i - \sin\theta_i}{\theta_i^3}, \quad \|\theta_i\| \geq \text{eps}, \\
                        \frac{1}{6}, \quad \|\theta_i\| < \text{eps},
                    \end{array}
                \right.

        .. math::
            C = 1
    
    Note:
        The detailed explanation of the above :math:`\mathrm{Exp}`: calculation can be found in the paper:

        * Grassia, F. Sebastian., `Practical Parameterization of Rotations using the
          Exponential Map <https://www.tandfonline.com/doi/pdf/10.1080/10867651.1998.10487493>`_,
          Journal of graphics tools, 1998.

        Assume we have a unit rotation axis :math:`\mathbf{n}~(\|\mathbf{n}\|=1)` and rotation angle
        :math:`\theta~(0\leq\theta<2\pi)`, let :math:`\mathbf{x}=\theta\mathbf{n}`, then the corresponding
        quaternion with unit norm :math:`\mathbf{q}` can be represented as:

            .. math::
                \mathbf{q} = \left[\frac{\sin(\theta/2)}{\theta} \mathbf{x}, \cos(\theta/2) \right].

        Given :math:`\mathbf{x}=\theta\mathbf{n}`, to find its corresponding quaternion
        :math:`\mathbf{q}`, we first calculate the rotation angle :math:`\theta` using:

            .. math::
                \theta = \|\mathbf{x}\|.

        Then, the corresponding quaternion is:
        
            .. math::
                \mathbf{q} = \left[\frac{\sin(\|\mathbf{x}\|/2)}{\|\mathbf{x}\|} \mathbf{x}, \cos(\|\mathbf{x}\|/2) \right].

        If :math:`\|\mathbf{x}\|` is small (:math:`\|\mathbf{x}\|\le \text{eps}`),
        we use the Taylor Expansion form of :math:`\sin(\|\mathbf{x}\|/2)` and :math:`\cos(\|\mathbf{x}\|/2)`.

        More details about :math:`^s\mathbf{W}_i` in :obj:`sim3_type` can be found in Eq. (5.7):

        * H. Strasdat, `Local Accuracy and Global Consistency for Efficient Visual
          SLAM <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.640.199&rep=rep1&type=pdf>`_,
          Dissertation. Department of Computing, Imperial College London, 2012.

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
                \mathbf{y}_i = 
                    \left\{
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
        :math:`\mathbf{J}_i` is the left Jacobian for :obj:`SO3_type` input.

    * If input :math:`\mathbf{x}`'s :obj:`ltype` is :obj:`RxSO3_type`
      (input :math:`\mathbf{x}` is an instance of :meth:`RxSO3`):

        Let :math:`\mathbf{q}_i`, :math:`s_i` be the rotation and scale parts of :math:`\mathbf{x}_i`, respectively;
        :math:`\mathbf{y}` be the output.

        .. math::
            \mathbf{y}_i = \left[\mathrm{Log}(\mathbf{q}_i), \log(s_i) \right].

    * If input :math:`\mathbf{x}`'s :obj:`ltype` is :obj:`Sim3_type` (input :math:`\mathbf{x}`
      is an instance of :meth:`Sim3`):

        Let :math:`\mathbf{t}_i`, :math:`^s\mathbf{q}_i` be the translation and :obj:`RxSO3` parts
        of :math:`\mathbf{x}_i`, respectively; :math:`\boldsymbol{\phi}_i`, :math:`\sigma_i` be the corresponding 
        Lie Algebra of the SO3 and scale part of :math:`^s\mathbf{q}_i`, :math:`\boldsymbol{\Phi}_i` be the skew 
        matrix of :math:`\boldsymbol{\phi}_i`, :math:`\boldsymbol{\phi}_i` can be represented as 
        :math:`\theta_i\mathbf{n}_i`, :math:`s_i = e^\sigma_i`, :math:`\mathbf{y}` be the output.

        .. math::
            \mathbf{y}_i = \left[^s\mathbf{W}_i^{-1}\mathbf{t}_i, \mathrm{Log}(^s\mathbf{q}_i) \right],

        where

            .. math::
               ^s\mathbf{W}_i = A\boldsymbol{\Phi}_i + B\boldsymbol{\Phi}_i^2 + C\mathbf{I}

        in which if :math:`\|\sigma_i\| > \text{eps}`:

        .. math::
            A = \left\{
                    \begin{array}{ll} 
                        \frac{s_i\sin\theta_i\sigma_i + (1-s_i\cos\theta_i)\theta_i}
                        {\theta_i(\sigma_i^2 + \theta_i^2)}, \quad \|\theta_i\| > \text{eps}, \\
                        \frac{(\sigma_i-1)s_i+1}{\sigma_i^2}, \quad \|\theta_i\| \leq \text{eps},
                    \end{array}
                \right.

        .. math::
            B = 
            \left\{
                \begin{array}{ll} 
                    \left( C - \frac{(s_i\cos\theta_i-1)\sigma+ s_i\sin\theta_i\sigma_i}
                    {\theta_i^2+\sigma_i^2}\right)\frac{1}{\theta_i^2}, \quad \|\theta_i\| > \text{eps}, \\
                    \frac{s_i\sigma_i^2/2 + s_i-1-\sigma_i s_i}{\sigma_i^3}, \quad \|\theta_i\| \leq \text{eps},
                \end{array}
            \right.

        .. math::
            C = \frac{e^{\sigma_i} - 1}{\sigma_i}\mathbf{I}

        otherwise:

        .. math::
            A = \left\{
                    \begin{array}{ll} 
                        \frac{1-\cos\theta_i}{\theta_i^2}, \quad \|\theta_i\| > \text{eps}, \\
                        \frac{1}{2}, \quad \|\theta_i\| \leq \text{eps},
                    \end{array}
                \right.

        .. math::
            B = \left\{
                    \begin{array}{ll} 
                        \frac{\theta_i - \sin\theta_i}{\theta_i^3}, \quad \|\theta_i\| > \text{eps}, \\
                        \frac{1}{6}, \quad \|\theta_i\| \leq \text{eps},
                    \end{array}
                \right.

        .. math::
            C = 1

    Note:
        The :math:`\mathrm{arctan}`-based Logarithm map implementation thanks to the paper:

        * C. Hertzberg et al., `Integrating Generic Sensor Fusion Algorithms with Sound State
          Representation through Encapsulation of Manifolds <https://doi.org/10.1016/j.inffus.2011.08.003>`_,
          Information Fusion, 2013.

        Assume we have a unit rotation axis :math:`\mathbf{n}` and rotation angle
        :math:`\theta~(0\leq\theta<2\pi)`, then the corresponding quaternion with
        unit norm :math:`\mathbf{q}` can be represented as

            .. math::
                \mathbf{q} = \left[\sin(\theta/2) \mathbf{n}, \cos(\theta/2) \right]

        Therefore, given a quaternion :math:`\mathbf{q}=[\boldsymbol{\nu}, w]`, where :math:`\boldsymbol{\nu}`
        is the vector part, :math:`w` is the scalar part, to find the corresponding rotation vector ,
        the rotation angle :math:`\theta` can be obtained as 

            .. math::
                \theta = 2\mathrm{arctan}(\|\boldsymbol{\nu}\|/w),~\|\boldsymbol{\nu}\| = \sin(\theta/2), 

        The unit rotation axis :math:`\mathbf{n}` can be obtained as :math:`\mathbf{n} =
        \frac{\boldsymbol{\nu}}{{\|\boldsymbol{\nu}\|}}`. Hence, the corresponding rotation vector is 

            .. math::
                \theta \mathbf{n} = 2\frac{\mathrm{arctan}
                (\|\boldsymbol{\nu}\|/w)}{\|\boldsymbol{\nu}\|}\boldsymbol{\nu}.

        More details about :math:`^s\mathbf{W}_i` in :obj:`Sim3_type` can be found in Eq. (5.7):

        * H. Strasdat, `Local accuracy and global consistency for efficient visual SLAM
          <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.640.199&rep=rep1&type=pdf>`_, 
          Dissertation. Department of Computing, Imperial College London, 2012.

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
        * - (:obj:`RxSO3_type`, :obj:`rxso3_type`)
          - :math:`(\mathcal{G}\in\mathbb{R}^{*\times5}, \mathcal{g}\in\mathbb{R}^{*\times4})`
          - :math:`\mapsto`
          - :math:`\mathcal{g}\in\mathbb{R}^{*\times4}`
          - :obj:`rxso3_type`

    Let the input be (:math:`\mathbf{x}`, :math:`\mathbf{p}`), :math:`\mathbf{y}` be the output.

        .. math::
            \mathbf{y}_i = \mathbf{J}^{-1}_i(\mathbf{x}_i)\mathbf{p}_i,

        where :math:`\mathbf{J}^{-1}_i(\mathbf{x}_i)` is the inverse of left Jacobian of :math:`\mathbf{x}_i`. 

    * If input (:math:`\mathbf{x}`, :math:`\mathbf{p}`)'s :obj:`ltype` are :obj:`SO3_type` and :obj:`so3_type`
      (input :math:`\mathbf{x}` is an instance of :meth:`SO3`, :math:`\mathbf{p}` is an instance of :meth:`so3`).
      Let :math:`\boldsymbol{\phi}_i = \theta_i\mathbf{n}_i` be the corresponding Lie Algebra of :math:`\mathbf{x}_i`, 
      :math:`\boldsymbol{\Phi}_i` be the skew matrix (:meth:`pypose.vec2skew`) of :math:`\boldsymbol{\phi}_i`:

        .. math::
            \mathbf{J}^{-1}_i(\mathbf{x}_i) = \mathbf{I} - \frac{1}{2}\boldsymbol{\Phi}_i +
            \mathrm{coef}\boldsymbol{\Phi}_i^2

      where :math:`\mathbf{I}` is the identity matrix with the same dimension as :math:`\boldsymbol{\Phi}_i`, and 

        .. math::
            \mathrm{coef} = \left\{
                                \begin{array}{ll} 
                                    \frac{1}{\theta_i^2} - \frac{\cos{\frac{\theta_i}{2}}}{2\theta\sin{\frac{\theta_i}{2}}},
                                    \quad \|\theta_i\| > \text{eps}, \\
                                    \frac{1}{12},
                                    \quad \|\theta_i\| \leq \text{eps}
                                \end{array}
                             \right.

    * If input (:math:`\mathbf{x}`, :math:`\mathbf{p}`)'s :obj:`ltype` are :obj:`SE3_type` and :obj:`se3_type`
      (input :math:`\mathbf{x}` is an instance of :meth:`SE3`, :math:`\mathbf{p}` is an instance of :meth:`se3`).
      Let :math:`\boldsymbol{\phi}_i = \theta_i\mathbf{n}_i` be the corresponding Lie Algebra of the SO3 part of 
      :math:`\mathbf{x}_i`, :math:`\boldsymbol{\tau}_i` be the Lie Algebra of the translation part of :math:`\mathbf{x}_i`; 
      :math:`\boldsymbol{\Phi}_i` and :math:`\boldsymbol{\Tau}_i` be the skew matrices, respectively:

        .. math::
            \mathbf{J}^{-1}_i(\mathbf{x}_i) = \left[
                                \begin{array}{cc} 
                                    \mathbf{J}_i^{-1}(\boldsymbol{\Phi}_i) & -\mathbf{J}_i^{-1}(\boldsymbol{\Phi}_i)
                                    \mathbf{Q}_i(\boldsymbol{\tau}_i, \boldsymbol{\phi}_i)\mathbf{J}_i^{-1}(\boldsymbol{\Phi}_i) \\
                                    \mathbf{0} & \mathbf{J}_i^{-1}(\boldsymbol{\Phi}_i)
                                \end{array}
                             \right]

        where :math:`\mathbf{J}_i^{-1}(\boldsymbol{\Phi}_i)` is the inverse of left Jacobian of the SO3 part of :math:`\mathbf{x}_i`.
        :math:`\mathbf{Q}_i(\boldsymbol{\tau}_i, \boldsymbol{\phi}_i)` is 

        .. math::
            \begin{align*}
                \mathbf{Q}_i(\boldsymbol{\tau}_i, \boldsymbol{\phi}_i) = \frac{1}{2}\boldsymbol{\Tau}_i &+ c_1
                (\boldsymbol{\Phi_i\Tau_i} + \boldsymbol{\Tau_i\Phi_i} + \boldsymbol{\Phi_i\Tau_i\Phi_i}) \\
                 &+ c_2 (\boldsymbol{\Phi_i^2\Tau_i} + \boldsymbol{\Tau_i\Phi_i^2} - 3\boldsymbol{\Phi_i\Tau_i\Phi_i})\\
                 &+ c_3 (\boldsymbol{\Phi_i\Tau_i\Phi_i^2} + \boldsymbol{\Phi_i^2\Tau_i\Phi_i})  
            \end{align*}

        where,

        .. math::
            c_1 = \left\{
                    \begin{array}{ll} 
                        \frac{\theta_i - \sin\theta_i}{\theta_i^3}, \quad \|\theta_i\| > \text{eps}, \\
                        \frac{1}{6}-\frac{1}{120}\theta_i^2,
                        \quad \|\theta_i\| \leq \text{eps}
                    \end{array}
                    \right.

        .. math::
            c_2 = \left\{
                    \begin{array}{ll} 
                        \frac{\theta_i^2 +2\cos\theta_i - 2}{2\theta_i^4}, \quad \|\theta_i\| > \text{eps}, \\
                        \frac{1}{24}-\frac{1}{720}\theta_i^2,
                        \quad \|\theta_i\| \leq \text{eps}
                    \end{array}
                    \right.

        .. math::
            c_3 = \left\{
                    \begin{array}{ll} 
                        \frac{2\theta_i - 3\sin\theta_i + \theta_i\cos\theta_i}{2\theta_i^5}, 
                        \quad \|\theta_i\| > \text{eps}, \\
                        \frac{1}{120}-\frac{1}{2520}\theta_i^2,
                        \quad \|\theta_i\| \leq \text{eps}
                    \end{array}
                    \right.           

    * If input (:math:`\mathbf{x}`, :math:`\mathbf{p}`)'s :obj:`ltype` are :obj:`Sim3_type` and :obj:`sim3_type`
      (input :math:`\mathbf{x}` is an instance of :meth:`Sim3`, :math:`\mathbf{p}` is an instance of :meth:`sim3`).
      The inverse of left Jacobian can be approximated as:

        .. math::
            \mathbf{J}^{-1}_i(\mathbf{x}_i) = \sum_{n=0}(-1)^n\frac{B_n}{n!}(\boldsymbol{\xi}_i^{\curlywedge})^n

        where :math:`B_n` is the Bernoulli number: :math:`B_0 = 1`, :math:`B_1 = -\frac{1}{2}`,
        :math:`B_2 = \frac{1}{6}`, :math:`B_3 = 0`, :math:`B_4 = -\frac{1}{30}`.
        :math:`\boldsymbol{\xi}_i^{\curlywedge} = \mathrm{adj}(\boldsymbol{\xi}_i^{\wedge})` and :math:`\mathrm{adj}` 
        is the adjoint of the Lie algebra :math:`\mathfrak{sim}(3)`, e.g., :math:`\boldsymbol{\xi}_i \in \mathfrak{sim}(3)`.
        Notice that if notate :math:`\boldsymbol{X}_i = \mathrm{Adj}(\mathbf{x}_i)` and :math:`\mathrm{Adj}` 
        is the adjoint of the Lie group :math:`\mathrm{Sim}(3)`, there is a nice property:
        :math:`\mathrm{Adj}(\mathrm{Exp}(\boldsymbol{\xi}_i^{\curlywedge})) = \mathrm{Exp}(\mathrm{adj}(\boldsymbol{\xi}_i^{\wedge}))`, 
        or :math:`\boldsymbol{X}_i = \mathrm{Exp}(\boldsymbol{\xi}_i^{\curlywedge})`.
        

    * If input (:math:`\mathbf{x}`, :math:`\mathbf{p}`)'s :obj:`ltype` are :obj:`RxSO3_type` and :obj:`rxso3_type`
      (input :math:`\mathbf{x}` is an instance of :meth:`RxSO3`, :math:`\mathbf{p}` is an instance of :meth:`rxso3`).
      Let :math:`\boldsymbol{\phi}_i` be the corresponding Lie Algebra of the SO3 part of
      :math:`\mathbf{x}_i`, :math:`\boldsymbol{\Phi}_i` be the skew matrix (:meth:`pypose.vec2skew`),
      The inverse of left Jacobian of :math:`\mathbf{x}_i` is the same as that for the SO3 part of :math:`\mathbf{x}_i`.

        .. math::
            \mathbf{J}^{-1}_i(\mathbf{x}_i) = \left[
                                \begin{array}{cc} 
                                    \mathbf{J}_i^{-1}(\boldsymbol{\Phi}_i) & \mathbf{0} \\
                                    \mathbf{0} & 1
                                \end{array}
                             \right]

        where :math:`\mathbf{J}_i^{-1}(\boldsymbol{\Phi}_i)` is the
        inverse of left Jacobian of the SO3 part of :math:`\mathbf{x}_i`.

    Note:
        :math:`\mathrm{Jinvp}` is usually used in the Baker-Campbell-Hausdorff formula
        (BCH formula) when performing LieTensor multiplication.
        One can refer to this paper for more details:

        * J. Sola et al., `A micro Lie theory for state estimation in
          robotics <https://arxiv.org/abs/1812.01537>`_, arXiv preprint arXiv:1812.01537 (2018).

        In particular, Eq. (146) is the math used in the :obj:`SO3_type`, :obj:`so3_type` scenario; 
        Eq. (179b) and Eq. (180) are the math used in the :obj:`SE3_type`, :obj:`se3_type` scenario.

        For Lie groups such as :obj:`Sim3_type`, :obj:`sim3_type`,
        there is no analytic expression for the left Jacobian and its inverse. 
        Numerical approximation is used based on series expansion.
        One can refer to Eq. (26) of this paper for more details about the approximation:

        * Z. Teed et al., `Tangent Space Backpropagation for 3D Transformation Groups.
          <https://arxiv.org/pdf/2103.12032.pdf>`_, in IEEE/CVF Conference on Computer Vision and
          Pattern Recognition (CVPR) (2021).

        In particular, the Bernoulli numbers can be obtained from Eq. (7.72) of this famous book:

        * T. Barfoot, `State Estimation for Robotics.
          <https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.708.1086&rep=rep1&type=pdf>`_,
          Cambridge University Press (2017).

    Example:

        * :math:`\mathrm{Jinvp}`: (:obj:`SO3`, :obj:`so3`) :math:`\mapsto` :obj:`so3`

        >>> x = pp.randn_SO3(2)
        >>> a = pp.randn_so3(2)
        >>> x.Jinvp(a) # equivalent to: pp.Jinvp(x, a)
            tensor([[-0.1068,  1.6051, -2.0121],
                    [-0.6726, -0.0345,  0.2493]])

        * :math:`\mathrm{Jinvp}`: (:obj:`SE3`, :obj:`se3`) :math:`\mapsto` :obj:`se3`

        >>> x = pp.randn_SE3(2)
        >>> a = pp.randn_se3(2)
        >>> x.Jinvp(a) # equivalent to: pp.Jinvp(x, a)
            tensor([[-1.3803,  0.7891, -0.4268,  0.6917, -0.2167,  0.3333],
                    [-1.4517, -0.8059,  0.9343,  1.7398,  0.6579,  0.4785]])

        * :math:`\mathrm{Jinvp}`: (:obj:`Sim3`, :obj:`sim3`) :math:`\mapsto` :obj:`sim3`

        >>> x = pp.randn_Sim3(2)
        >>> a = pp.randn_sim3(2)
        >>> x.Jinvp(a) # equivalent to: pp.Jinvp(x, a)
            tensor([[ 0.3943, -1.2546,  0.3209,  0.2298, -1.1028, -1.4039,  0.3704],
                    [-0.3591,  0.4190,  0.2833, -0.3121,  1.6293, -0.8617, -0.7911]])

        * :math:`\mathrm{Jinvp}`: (:obj:`RxSO3`, :obj:`rxso3`) :math:`\mapsto` :obj:`rxso3`

        >>> x = pp.randn_RxSO3(2)
        >>> a = pp.randn_rxso3(2)
        >>> x.Jinvp(a) # equivalent to: pp.Jinvp(x, a)
            tensor([[ 0.1730, -1.3778,  0.1657,  0.1820],
                    [-1.0347,  1.6627,  0.3992,  0.1227]])
    """
    return input.Jinvp(p)


@assert_ltype
def Jr(x):
    r"""
    The batched right Jacobian of a LieTensor.

    Args:
        input (LieTensor): the input LieTensor (either Lie Group or Lie Algebra)

    Return:
        Tensor: the right Jocobian Matrices

    Example:

    * :math:`\mathrm{Jr}`: :meth:`so3` :math:`\mapsto` :math:`\mathcal{R}^{*\times 3\times 3}`

    >>> x = pp.randn_so3(requires_grad=True)
    >>> x.Jr()
    tensor([[ 0.9289, -0.3053, -0.0895],
            [ 0.3180,  0.9082,  0.1667],
            [ 0.0104, -0.1889,  0.9757]], grad_fn=<SWhereBackward0>)
    """
    return x.Jr()
