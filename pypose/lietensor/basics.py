import math
import torch
'''
This basics file includes functions needed to implement LieTensor.
'''

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
                & = \left.\frac{\partial\mathrm{Log}(f(\mathrm{Exp}(\bm{\tau})\times\mathcal{X}))
                    \times f(\mathcal{X})^{-1})}{\partial \bm{\tau}}\right|_{\bm{\tau=\bm{0}}}
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
    Multiply input LieTensor by other.

    .. math::
        \bm{y}_i =
        \bm{x}_i \ast \bm{a}_i

    where :math:`\bm{x}` is the ``input`` LieTensor, :math:`\bm{a}` is the ``other`` Tensor or
    LieTensor, and :math:`\bm{y}` is the output.

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

    Note:
        - When ``other`` is a Tensor, this operator is equivalent to :meth:`Act`.
        - When ``other`` is a number and ``input`` is a Lie Algebra, this operator performs
          simple element-wise multiplication.
        - When ``input`` is a Lie Group, more details are shown below.
   
    * Input :math:`\bm{x}`'s :obj:`ltype` is :obj:`SO3_type`
      (input :math:`\bm{x}` is an instance of :meth:`SO3`):

      .. math::
        {\displaystyle \bm{y}_i={\begin{bmatrix}
              q_i^wq_i^{w'} - q_i^xq_i^{x'} -q_i^yq_i^{y'} - q_i^zq_i^{z'}\\
              q_i^wq_i^{x'} + q_i^xq_i^{w'} +q_i^yq_i^{z'} - q_i^zq_i^{y'}\\
              q_i^wq_i^{y'} - q_i^xq_i^{z'} +q_i^yq_i^{w'} + q_i^zq_i^{x'}\\
              q_i^wq_i^{z'} + q_i^xq_i^{y'} -q_i^yq_i^{x'} + q_i^zq_i^{w'}
              \end{bmatrix}
        }^T},
    
      where :math:`\bm{x}_i = [q_i^x, q_i^y, q_i^z, q_i^w]` and
      :math:`\bm{a}_i = [q_i^{x'}, q_i^{y'}, q_i^{z'}, q_i^{w'}]` are the ``input``
      and ``other`` LieTensor, respectively.
        
    Note:
        :math:`\mathbf{y}_i` can be simply derived by taking the complex number multiplication.

        .. math:: 
            \bm{y}_i = 
            (q_i^x\mathbf{i} + q_i^y\mathbf{j} + q_i^z\mathbf{k} + q_i^w) \ast 
            (q_i^{x'}\mathbf{i} + q_i^{y'}\mathbf{j} + q_i^{z'}\mathbf{k} + q_i^{w'}),

        where and :math:`\mathbf{i}` :math:`\mathbf{j}`, and :math:`\mathbf{k}` are the imaginary
        units.

        .. math::
            \mathbf{i}^2 = \mathbf{j}^2 = \mathbf{k}^2 = \mathbf{ijk} = -1

    * Input :math:`\bm{x}`'s :obj:`ltype` is :obj:`SE3_type`
      (input :math:`\bm{x}` is an instance of :meth:`SE3`):

        .. math::
            \bm{y}_i = [\mathbf{q}_i * \mathbf{t}_i' + \mathbf{t}_i,
                        \mathbf{q}_i * \mathbf{q}_i']
        
        where :math:`\bm{x}_i = [\mathbf{t}_i, \mathbf{q}_i]` and
        :math:`\bm{a}_i = [\mathbf{t}_i', \mathbf{q}_i']` are the ``input`` and ``other``
        LieTensor, respectively; :math:`\mathbf{t}_i`, :math:`\mathbf{t}_i'` and
        :math:`\mathbf{q}_i`, :math:`\mathbf{q}_i'` are their translation and :obj:`SO3`
        parts, respectively; the operator :math:`\ast` denotes the obj:`SO3_type`
        multiplication introduced above.
            
    * Input :math:`\bm{x}`'s :obj:`ltype` is :obj:`RxSO3_type`
      (input :math:`\bm{x}` is an instance of :meth:`RxSO3`)

        .. math::
            \bm{y}_i = [\mathbf{q}_i * \mathbf{q}_i', s_is_i']

        where :math:`s_i` and :math:`s_i'` are the scale parts of the ``input`` and
        ``other`` LieTensor, respectively.

    * Input :math:`\bm{x}`'s :obj:`ltype` is :obj:`Sim3_type`
      (input :math:`\bm{x}` is an instance of :meth:`Sim3`):

        .. math::
            \bm{y}_i = [\mathbf{q}_i * \mathbf{t}_i' + \mathbf{t}_i,
                            \mathbf{q}_i * \mathbf{q}_i', s_is_i'] 

        where :math:`\bm{x}_i = [\mathbf{t}_i, \mathbf{q}_i, s_i]` and
        :math:`\bm{a}_i = [\mathbf{t}_i', \mathbf{q}_i', s_i']` are the ``input`` and
        ``other`` LieTensor, respectively.

    Examples:
        * :obj:`Lie Algebra` :math:`*` :obj:`Number` :math:`\mapsto` :obj:`Lie Algebra`

            >>> x = pp.randn_so3()
            >>> x
            so3Type LieTensor:
            LieTensor([ 0.3018, -1.0246,  0.7784])
            >>> a = 5
            >>> # The following two operations are equivalent.
            >>> x * a
            so3Type LieTensor:
            LieTensor([ 1.5090, -5.1231,  3.8919])
            >>> pp.mul(x, 5)
            so3Type LieTensor:
            LieTensor([ 1.5090, -5.1231,  3.8919])

        * :obj:`Lie Group` :math:`*` :obj:`Tensor` :math:`\mapsto` :obj:`Tensor`

            >>> x = pp.randn_SO3()
            >>> a = torch.randn(3)
            >>> x, a
            (SO3Type LieTensor:
            LieTensor([ 0.6047, -0.2129, -0.1781,  0.7465]),
            tensor([-0.1811, -0.2278, -1.9956]))
            >>> x * a
            tensor([ 0.9089,  1.6984, -0.5969])
            >>> a = torch.randn(4)
            >>> a 
            tensor([ 1.5236, -1.2757, -0.7140,  0.2467])
            >>> x * a
            tensor([ 1.6588, -0.4687, -1.2196,  0.2467]

        * :obj:`SO3_type` :math:`*` :obj:`SO3_type` :math:`\mapsto` :obj:`SO3_type`

            >>> a = pp.randn_SO3()
            >>> a
            SO3Type LieTensor:
            LieTensor([ 0.0118, -0.7042, -0.4516,  0.5478])
            >>> x * a
            SO3Type LieTensor:
            LieTensor([ 0.3108, -0.3714, -0.8579,  0.1715])

        * :obj:`SE3_type` :math:`*` :obj:`SE3_type` :math:`\mapsto` :obj:`SE3_type`

            >>> x = pp.randn_SE3()
            >>> a = pp.randn_SE3()
            >>> x, a
            (SE3Type LieTensor:
            LieTensor([ 0.7819,  1.8541, -0.2857, -0.1970,  0.4742,  0.1109,  0.8509]),
            SE3Type LieTensor:
            LieTensor([ 0.6039, -1.4076,  0.3496,  0.7297,  0.3971,  0.2849,  0.4783]))
            >>> x * a
            SE3Type LieTensor:
            LieTensor([ 1.8949,  0.7456, -0.3104,  0.6177,  0.7017, -0.1287,  0.3308])

        * :obj:`RxSO3_type` :math:`*` :obj:`RxSO3_type` :math:`\mapsto` :obj:`RxSO3_type`

            >>> x = pp.randn_RxSO3()
            >>> a = pp.randn_RxSO3()
            >>> x, a
            (RxSO3Type LieTensor:
            LieTensor([-0.7518, -0.6481,  0.0933, -0.0775,  1.5791]),
            RxSO3Type LieTensor:
            LieTensor([ 0.2757,  0.3102, -0.4086,  0.8129,  0.6593]))
            >>> x * a
            RxSO3Type LieTensor:
            LieTensor([-0.3967, -0.8323,  0.0530,  0.3835,  1.0411])

        * :obj:`Sim3_type` :math:`*` :obj:`Sim3_type` :math:`\mapsto` :obj:`Sim3_type`

            >>> x = pp.randn_Sim3()
            >>> a = pp.randn_Sim3()
            >>> x, a
            (Sim3Type LieTensor:
            LieTensor([-0.3439, -0.2309, -0.6571,  0.3170, -0.6594, -0.1100,  0.6728, 0.6296]),
            Sim3Type LieTensor:
            LieTensor([-0.7434,  1.8613, -2.1315,  0.7688, -0.0268,  0.0520,  0.6367, 1.7745]))
            >>> x * a
            Sim3Type LieTensor:
            LieTensor([ 0.5740,  1.3197, -0.2752,  0.6819, -0.5389,  0.4634,  0.1727, 1.1172])
    '''
    return input * other
