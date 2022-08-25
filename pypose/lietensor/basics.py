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


def mul(input, other):
    r'''
    Multiplies input LieTensor by other.

    .. math::
        \bm{y}_i =
        \bm{x}_i \ast \bm{a}_i

    where :math:`\bm{x}` is the ``input`` LieTensor, :math:`\bm{a}` is the ``other`` value,
    and :math:`\bm{y}` is the output value.

    Args:
        input (:obj:`LieTensor`): the input LieTensor (Lie Group or Lie Algebra).

        other (:obj:`Number`, :obj:`Tensor`, or :obj:`LieTensor`): the value for input to be
            multiplied by.

    Return:
        :obj:`Tensor`/:obj:`LieTensor`: the product of ``input`` and ``other``.

    .. list-table:: List of :obj:`pypose.mul` cases 
        :widths: 25 30 25
        :header-rows: 1

        * - input :obj:`LieTensor`
          - other
          - output
        * - Lie Algebra
          - :obj:`Number`
          - :obj:`Lie Algebra`
        * - Lie Group
          - :obj:`Tensor` :math:`\in \mathbb{R^{*\times3}}`
          - :obj:`Tensor`
        * - Lie Group
          - :obj:`Tensor` :math:`\in \mathbb{R^{*\times4}}`
          - :obj:`Tensor`
        * - Lie Group
          - :obj:`Lie Group` 
          - :obj:`Lie Group`

    When multiplying a Lie Group by another Lie Group, they must have the 
    same Lie type.

    See :obj:`Act()` for multiplying by a Tensor.

    For multpilying by a :obj:`Lie Group`, see below:
   
    * Input :math:`\bm{x}`'s :obj:`ltype` is :obj:`SO3_type`
      (input :math:`\bm{x}` is an instance of :meth:`SO3`):

        .. math::
            q_i = [q_i^x, q_i^y, q_i^z, q_i^w],
            
        .. math::
            \bm{x}_i = [q_i] = q_i^xi + q_i^yj + q_i^zk + q_i^w,
        .. math::
            \bm{a}_i = [q_i'] = q_i^{x'}i + q_i^{y'}j + q_i^{z'}k + q_i^{w'},

        and :math:`i`, :math:`j`, :math:`k`, and :math:`1` 
        represent the standard basis of quaternions and 

        .. math::
            {i}^2 = {j}^2 = {k}^2 = {ijk} = -1,

        Using these definitions, the product of these quaternions is
        
        .. math:: 
            \bm{y}_i = 
            (q_i^xi + q_i^yj + q_i^zk + q_i^w) \ast (q_i^{x'}i + q_i^{y'}j + q_i^{z'}k + q_i^{w'}),

        Using the Hamilton product, we have the following

        .. math::
            {\displaystyle \bm{y}_i={\begin{bmatrix}
              q_i^wq_i^{w'} & -q_i^xq_i^{x'} & -q_i^yq_i^{y'} & -q_i^zq_i^{z'}\\
              q_i^wq_i^{x'} & q_i^xq_i^{w'} & q_i^yq_i^{z'} & -q_i^zq_i^{y'}\\
              q_i^wq_i^{y'} & -q_i^xq_i^{z'} & q_i^yq_i^{w'} & q_i^zq_i^{x'}\\
              q_i^wq_i^{z'} & q_i^xq_i^{y'} & -q_i^yq_i^{x'} & q_i^zq_i^{w'}
              \end{bmatrix}
            }}

    * Input :math:`\bm{x}`'s :obj:`ltype` is :obj:`SE3_type`
      (input :math:`\bm{x}` is an instance of :meth:`SE3`):

        .. math:: 
            t_i = [t_i^x, t_i^y, t_i^z],
        .. math::
            \bm{x}_i = [t_i, q_i],
        .. math::
            \bm{a}_i = [t_i', q_i'],

        Peforms same calculations as with :obj:`SO3_type` to calculate the 
        quaternion of the product, and, calculates the translational vector 
        with the method below,

        .. math::
            \bm{y}_i = [q_i \ast t_i' + t_i, q_i * q_i']
            
    * Input :math:`\bm{x}`'s :obj:`ltype` is :obj:`RxSO3_type`
      (input :math:`\bm{x}` is an instance of :meth:`RxSO3`)

        .. math::
            \bm{x}_i = [q_i, s_i]
        .. math::
            \bm{a}_i = [q_i', s_i']

        Peforms same calculations as with :obj:`SO3_type` to calculate the 
        quaternion of the product 

        .. math::
            \bm{y}_i = [q_i \ast q_i', s_is_i']

    * Input :math:`\bm{x}`'s :obj:`ltype` is :obj:`Sim3_type`
      (input :math:`\bm{x}` is an instance of :meth:`Sim3`):

        .. math::
            \bm{x}_i = [t_i, q_i, s_i],
            \bm{a}_i = [t_i', q_i', s_i']

        Peforms same calculations as with :obj:`RxSO3_type` to calculate the 
        quaternion and scaling factor of the product, and uses same :obj:`SE3_type` 
        calculations for the translational vector.

        .. math::
            \bm{y}_i = [q_i \ast t_i' + t_i, q_i \ast q_i', s_is_i'] 

    Examples:
        The following operations are equivalent.

        >>> x = pp.randn_so3()
        >>> x
        so3Type LieTensor:
        LieTensor([ 0.3018, -1.0246,  0.7784])
        >>> a = 5
        >>> x * a
        so3Type LieTensor:
        LieTensor([ 1.5090, -5.1231,  3.8919])
        >>> pp.mul(x, 5)
        so3Type LieTensor:
        LieTensor([ 1.5090, -5.1231,  3.8919])

        * :obj:`Lie Algebra` :math:`*` :obj:`Number` :math:`=` :obj:`Lie Algebra`

            >>> x * a
            so3Type LieTensor:
            LieTensor([ 1.5090, -5.1231,  3.8919])

        * :obj:`Lie Group` :math:`*` :obj:`Tensor` :math:`\in \mathbb{R^{*\times3}}`
          :math:`=` :obj:`Tensor`

            >>> x = pp.randn_SO3()
            >>> a = torch.randn(3)
            >>> x, a
            (SO3Type LieTensor:
            LieTensor([0.0092, 0.3450, 0.7255, 0.5954]), tensor([ 2.2862,  0.8660, -1.3799]))
            >>> x * a
            >>> tensor([-1.9929,  1.2682, -1.5167])

        * :obj:`Lie Group` :math:`*` :obj:`Tensor` :math:`\in \mathbb{R^{*\times4}}`
          :math:`=` :obj:`Tensor`

            >>> a = torch.randn(4)
            >>> a
            tensor([-1.4565,  0.3828, -0.6383,  1.4504])
            >>> x * a
            >>> tensor([-0.1755, -1.6004,  0.2886,  1.4504])

        * :obj:`LieTensor` :math:`*` :obj:`LieTensor` :math:`=` :obj:`LieTensor`

            >>> a = pp.randn_SO3()
            >>> a
            SO3Type LieTensor:
            LieTensor([-0.6754, -0.2603,  0.3311,  0.6053])
            >>> x * a
            SO3Type LieTensor:
            LieTensor([-0.0935, -0.4392,  0.8670,  0.2162])
    '''
    return input * other


def matmul(input, other):
    r'''
    Performs matrix multiplication with input LieTensor and other

    .. math::
        \bm{y}_i =
        \bm{x}_i \times \bm{a}_i

    where :math:`\bm{x}` is the ``input`` LieTensor, :math:`\bm{a}` is the ``other`` value,
    and :math:`\bm{y}` is the output value.

    Args:
        input (:obj:`LieTensor`): the input LieTensor (Lie Group or Lie Algebra).

        other (:obj:`Tensor` or :obj:`LieTensor`): the value for input to be multiplied by.

    Return:
        :obj:`Tensor`/:obj:`LieTensor`: the product of ``input`` and ``other``.

    .. list-table:: List of :obj:`pypose.matmul` cases 
        :widths: 25 30 25
        :header-rows: 1

        * - input :obj:`LieTensor`
          - other
          - output
        * - Lie Group
          - :obj:`Tensor` :math:`\in \mathbb{R^{*\times3}}`
          - :obj:`Tensor`
        * - Lie Group
          - :obj:`Tensor` :math:`\in \mathbb{R^{*\times4}}`
          - :obj:`Tensor`
        * - Lie Group
          - :obj:`Lie Group (input.ltype)` 
          - :obj:`Lie Group`

    See :obj:`Act()` for multiplying by a Tensor.

    For multpilying by a :obj:`Lie Group`, see below:
   
    * Input :math:`\bm{x}`'s :obj:`ltype` is :obj:`SO3_type`
      (input :math:`\bm{x}` is an instance of :meth:`SO3`):

        .. math::
            \bm{x} = [q_x, q_y, q_z, q_w] = q_xi + q_yj + q_zk + q_w,
        .. math::
            \bm{a} = [q_x', q_y', q_z', q_w'] = q_x'i + q_y'j + q_z'k + q_w',

        where :math:`\bm{i}`, :math:`\bm{j}`, :math:`\bm{k}`, and :math:`\bm{1}` 
        represent the standard basis of quaternions and 

        .. math::
            {i}^2 = {j}^2 = {k}^2 = {ijk} = -1,

        Using these definitions, the product of these quaternions is
        
        .. math:: 
            \bm{x} \times \bm{a} = 
            (q_xi + q_yj + q_zk + q_w) \times (q_x'i + q_y'j + q_z'k + q_w'),
        
        And rearranging terms simplifies this product to

        .. math::
            (\bm{x} \times \bm{a})_{xyz} = 
            q_{xyz} \times q_{xyz}' + q_wq_{xyz}' + q_w'q_{xyz},
        .. math::
            (\bm{x} \times \bm{a})_{w} = 
            q_wq_w' - q_{xyz} \cdot q_{xyz}',
        .. math::
            \bm{x} \times \bm{a} = 
            q_wq_w' - q_{xyz} \cdot q_{xyz}' + q_wq_{xyz}' + q_w'q_{xyz} + 
            q_{xyz} \times q_{xyz}'

        Alternatively, using the Hamilton product, we have the following

        .. math::
            (\bm{x} \times \bm{a})_{q_w} = q_wq_w' - q_xq_x' - q_yq_y' - q_zq_z',
        .. math::
            (\bm{x} \times \bm{a})_{q_x} = q_wq_x' + q_xq_w' + q_yq_z' - q_zq_y',
        .. math::
            (\bm{x} \times \bm{a})_{q_y} = q_wq_y' - q_xq_z' + q_yq_w' + q_zq_x,
        .. math::
            (\bm{x} \times \bm{a})_{q_z} = q_wq_z' + q_xq_y' - q_yq_x' + q_zq_w',

    * Input :math:`\bm{x}`'s :obj:`ltype` is :obj:`SE3_type`
      (input :math:`\bm{x}` is an instance of :meth:`SE3`):

        .. math::
            \bm{x} = [t_x, t_y, t_z, q_x, q_y, q_z, q_w],
        .. math::
            \bm{a} = [t_x', t_y', t_z', q_x', q_y', q_z', q_w'],

        Peforms same calculations as with :obj:`SO3_type` to calculate the 
        quaternion of the product :math:`(\bm{x} \times \bm{a})_{q}`, and, 
        calculates the translational vector with the method below,

        .. math::
            \bm{x}_{t} = [t_x, t_y, t_z],
        .. math::
            \bm{a}_{t} = [t_x', t_y', t_z'],
        .. math::
            \bm{x}_{q} = [q_x, q_y, q_z, q_w],
        .. math::
            (\bm{x} \times \bm{a})_{t} = \bm{x}_q \times \bm{a}_t + \bm{x}_t
            
    * Input :math:`\bm{x}`'s :obj:`ltype` is :obj:`RxSO3_type`
      (input :math:`\bm{x}` is an instance of :meth:`RxSO3`)

        .. math::
            \bm{x} = [q_x, q_y, q_z, q_w, s],
        .. math::
            \bm{a} = [q_x', q_y', q_z', q_w', s'],

        Peforms same calculations as with :obj:`SO3_type` to calculate the 
        quaternion of the product :math:`(\bm{x} \times \bm{a})_{q}`, and 

        .. math::
            (\bm{x} \times \bm{a})_{s} = ss'

    * Input :math:`\bm{x}`'s :obj:`ltype` is :obj:`Sim3_type`
      (input :math:`\bm{x}` is an instance of :meth:`Sim3`):

        .. math::
            \bm{x} = [t_x, t_y, t_z, q_x, q_y, q_z, q_w, s],
        .. math::
            \bm{a} = [t_x', t_y', t_z', q_x', q_y', q_z', q_w', s'],
        
        Based off of the calculations for the previous types, we have the following

        .. math::
            (\bm{x} \times \bm{a})_{q_w} = q_wq_w' - q_xq_x' - q_yq_y' - q_zq_z',
        .. math::
            (\bm{x} \times \bm{a})_{q_x} = q_wq_x' + q_xq_w' + q_yq_z' - q_zq_y',
        .. math::
            (\bm{x} \times \bm{a})_{q_y} = q_wq_y' - q_xq_z' + q_yq_w' + q_zq_x,
        .. math::
            (\bm{x} \times \bm{a})_{q_z} = q_wq_z' + q_xq_y' - q_yq_x' + q_zq_w',
        .. math::
            (\bm{x} \times \bm{a})_{s} = ss'
        .. math::
            (\bm{x} \times \bm{a})_{t} = \bm{x}_q \times \bm{a}_t + \bm{x}_t

    Examples:
        The following operations are equivalent.

        >>> x = pp.randn_so3()
        >>> a = pp.randn_so3()
        >>> x, a
        (so3Type LieTensor:
        LieTensor([ 0.6835,  0.6479, -1.0084]), so3Type LieTensor:
        LieTensor([-2.7218,  0.2900, -0.2014]))
        >>> x @ a
        so3Type LieTensor:
        LieTensor([-1.8603,  0.1879,  0.2031])
        >>> pp.matmul(x, a)
        so3Type LieTensor:
        LieTensor([-1.8603,  0.1879,  0.2031])

        * :obj:`Lie Group` @ :obj:`Tensor` :math:`\in \mathbb{R^{*\times3}}`
          :math:`=` :obj:`Tensor`

            >>> x = pp.randn_SO3()
            >>> a = torch.randn(3)
            >>> x, a
            (SO3Type LieTensor:
            LieTensor([-0.1576,  0.1065,  0.0651,  0.9796]), tensor([-1.0442,  0.2388,  1.0111]))
            >>> x @ a
            tensor([-0.8599,  0.4530,  1.1068])

        * :obj:`Lie Group` @ :obj:`Tensor` :math:`\in \mathbb{R^{*\times4}}`
          :math:`=` :obj:`Tensor`

            >>> a = torch.randn(4)
            >>> a 
            tensor([-1.4010, -0.3532,  0.6713, -0.3036])
            >>> x @ a
            tensor([-1.1741, -0.2477,  1.0479, -0.3036])
    '''
    return input @ other


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
