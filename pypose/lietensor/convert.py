import torch, warnings
from .utils import SO3, so3, SE3, RxSO3, Sim3
from .lietensor import LieTensor, SE3_type, SO3_type, Sim3_type, RxSO3_type


def mat2SO3(mat, check=True, rtol=1e-5, atol=1e-5):
    r"""Convert batched rotation or transformation matrices to SO3Type LieTensor.

    Args:
        mat (Tensor): the batched matrices to convert. If input is of shape :obj:`(*, 3, 4)`
            or :obj:`(*, 4, 4)`, only the top left 3x3 submatrix is used.
        check (bool, optional): flag to check if the input is valid rotation matrices (orthogonal
            and with a determinant of one). Set to ``False`` if less computation is needed.
            Default: ``True``.
        rtol (float, optional): relative tolerance when check is enabled. Default: 1e-05
        atol (float, optional): absolute tolerance when check is enabled. Default: 1e-05

    Return:
        LieTensor: the converted SO3Type LieTensor.

    Shape:
        Input: :obj:`(*, 3, 3)` or :obj:`(*, 3, 4)` or :obj:`(*, 4, 4)`

        Output: :obj:`(*, 4)`

    Let the input be matrix :math:`\mathbf{R}`, :math:`\mathbf{R}_i` represents each individual
    matrix in the batch. :math:`\mathbf{R}^{m,n}_i` represents the :math:`m^{\mathrm{th}}` row
    and :math:`n^{\mathrm{th}}` column of :math:`\mathbf{R}_i`, :math:`m,n\geq 1`, then the 
    quaternion can be computed by:

    .. math::
        \left\{\begin{aligned}
        q^x_i &= \mathrm{sign}(\mathbf{R}^{2,3}_i - \mathbf{R}^{3,2}_i) \frac{1}{2}
            \sqrt{1 + \mathbf{R}^{1,1}_i - \mathbf{R}^{2,2}_i - \mathbf{R}^{3,3}_i}\\
        q^y_i &= \mathrm{sign}(\mathbf{R}^{3,1}_i - \mathbf{R}^{1,3}_i) \frac{1}{2}
            \sqrt{1 - \mathbf{R}^{1,1}_i + \mathbf{R}^{2,2}_i - \mathbf{R}^{3,3}_i}\\
        q^z_i &= \mathrm{sign}(\mathbf{R}^{1,2}_i - \mathbf{R}^{2,1}_i) \frac{1}{2}
            \sqrt{1 - \mathbf{R}^{1,1}_i - \mathbf{R}^{2,2}_i + \mathbf{R}^{3,3}_i}\\
        q^w_i &= \frac{1}{2} \sqrt{1 + \mathbf{R}^{1,1}_i + \mathbf{R}^{2,2}_i + \mathbf{R}^{3,3}_i}
        \end{aligned}\right.,
    
    In summary, the output LieTensor should be of format:

    .. math::
        \textbf{y}_i = [q^x_i, q^y_i, q^z_i, q^w_i]

    Warning:
        Numerically, a transformation matrix is considered legal if:

        .. math::
            |{\rm det}(\mathbf{R}) - 1| \leq \texttt{atol} + \texttt{rtol}\times 1\\
            |\mathbf{RR}^{T} - \mathbf{I}| \leq \texttt{atol} + \texttt{rtol}\times \mathbf{I}
        
        where :math:`|\cdot |` is element-wise absolute function. When ``check`` is set to ``True``,
        illegal input will raise a ``ValueError``. Otherwise, no data validation is performed. 
        Illegal input will output an irrelevant result, which likely contains ``nan``.
        
    Examples:

        >>> input = torch.tensor([[0., -1.,  0.],
        ...                       [1.,  0.,  0.],
        ...                       [0.,  0.,  1.]])
        >>> pp.mat2SO3(input)
        SO3Type LieTensor:
        tensor([0.0000, 0.0000, 0.7071, 0.7071])

    See :meth:`pypose.SO3` for more details of the output LieTensor format.
    """

    if not torch.is_tensor(mat):
        mat = torch.tensor(mat)

    if len(mat.shape) < 2:
        raise ValueError("Input size must be at least 2 dimensions. Got {}".format(mat.shape))

    if not (mat.shape[-2:] == (3, 3) or mat.shape[-2:] == (3, 4) or mat.shape[-2:] == (4, 4)):
        raise ValueError("Input size must be a * x 3 x 3 or * x 3 x 4 or * x 4 x 4 tensor. \
                Got {}".format(mat.shape))

    mat = mat[..., :3, :3]
    shape = mat.shape

    with torch.no_grad():
        if check:
            e0 = mat @ mat.mT
            e1 = torch.eye(3, dtype=mat.dtype, device=mat.device)
            if not torch.allclose(e0, e1.expand_as(e0), rtol=rtol, atol=atol):
                raise ValueError("Input rotation matrices are not all orthogonal matrix")

            ones = torch.ones(shape[:-2], dtype=mat.dtype, device=mat.device)
            if not torch.allclose(torch.det(mat), ones, rtol=rtol, atol=atol):
                raise ValueError("Input rotation matrices' determinant are not all equal to 1")

    rmat_t = mat.mT

    mask_d2 = rmat_t[..., 2, 2] < atol

    mask_d0_d1 = rmat_t[..., 0, 0] > rmat_t[..., 1, 1]
    mask_d0_nd1 = rmat_t[..., 0, 0] < -rmat_t[..., 1, 1]

    t0 = 1 + rmat_t[..., 0, 0] - rmat_t[..., 1, 1] - rmat_t[..., 2, 2]
    q0 = torch.stack([rmat_t[..., 1, 2] - rmat_t[..., 2, 1],
                      t0, rmat_t[..., 0, 1] + rmat_t[..., 1, 0],
                      rmat_t[..., 2, 0] + rmat_t[..., 0, 2]], -1)
    t0_rep = t0.unsqueeze(-1).repeat((len(list(shape))-2)*(1,)+(4,))

    t1 = 1 - rmat_t[..., 0, 0] + rmat_t[..., 1, 1] - rmat_t[..., 2, 2]
    q1 = torch.stack([rmat_t[..., 2, 0] - rmat_t[..., 0, 2],
                      rmat_t[..., 0, 1] + rmat_t[..., 1, 0],
                      t1, rmat_t[..., 1, 2] + rmat_t[..., 2, 1]], -1)
    t1_rep = t1.unsqueeze(-1).repeat((len(list(shape))-2)*(1,)+(4,))

    t2 = 1 - rmat_t[..., 0, 0] - rmat_t[..., 1, 1] + rmat_t[..., 2, 2]
    q2 = torch.stack([rmat_t[..., 0, 1] - rmat_t[..., 1, 0],
                      rmat_t[..., 2, 0] + rmat_t[..., 0, 2],
                      rmat_t[..., 1, 2] + rmat_t[..., 2, 1], t2], -1)
    t2_rep = t2.unsqueeze(-1).repeat((len(list(shape))-2)*(1,)+(4,))

    t3 = 1 + rmat_t[..., 0, 0] + rmat_t[..., 1, 1] + rmat_t[..., 2, 2]
    q3 = torch.stack([t3, rmat_t[..., 1, 2] - rmat_t[..., 2, 1],
                      rmat_t[..., 2, 0] - rmat_t[..., 0, 2],
                      rmat_t[..., 0, 1] - rmat_t[..., 1, 0]], -1)
    t3_rep = t3.unsqueeze(-1).repeat((len(list(shape))-2)*(1,)+(4,))

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.unsqueeze(-1).type_as(q0)
    mask_c1 = mask_c1.unsqueeze(-1).type_as(q1)
    mask_c2 = mask_c2.unsqueeze(-1).type_as(q2)
    mask_c3 = mask_c3.unsqueeze(-1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= 2*torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa

    q = q.view(shape[:-2]+(4,))
    # wxyz -> xyzw
    q = q.index_select(-1, torch.tensor([1, 2, 3, 0], device=q.device))

    return SO3(q)


def mat2SE3(mat, check=True, rtol=1e-5, atol=1e-5):
    r"""Convert batched rotation or transformation matrices to SE3Type LieTensor.

    Args:
        mat (Tensor): the batched matrices to convert. If input is of shape :obj:`(*, 3, 3)`, then
            translation will be filled with zero. For input with shape :obj:`(*, 3, 4)`, the last
            row will be treated as ``[0, 0, 0, 1]``.
        check (bool, optional): flag to check if the input is valid rotation matrices (orthogonal
            and with a determinant of one). Set to ``False`` if less computation is needed.
            Default: ``True``.
        rtol (float, optional): relative tolerance when check is enabled. Default: 1e-05
        atol (float, optional): absolute tolerance when check is enabled. Default: 1e-05

    Return:
        LieTensor: the converted SE3Type LieTensor.

    Shape:
        Input: :obj:`(*, 3, 3)` or :obj:`(*, 3, 4)` or :obj:`(*, 4, 4)`

        Output: :obj:`(*, 7)`

    Let the input be matrix :math:`\mathbf{T}`,  :math:`\mathbf{T}_i` represents each individual
    matrix in the batch. :math:`\mathbf{R}_i\in\mathbb{R}^{3\times 3}` be the top left 3x3 block 
    matrix of :math:`\mathbf{T}_i`. :math:`\mathbf{T}^{m,n}_i` represents the :math:`m^{\mathrm{th}}` 
    row and :math:`n^{\mathrm{th}}` column of :math:`\mathbf{T}_i`, :math:`m,n\geq 1`, then the 
    translation and quaternion can be computed by:

    .. math::
        \left\{\begin{aligned}
        t^x_i &= \mathbf{T}^{1,4}_i\\
        t^y_i &= \mathbf{T}^{2,4}_i\\
        t^z_i &= \mathbf{T}^{3,4}_i\\
        q^x_i &= \mathrm{sign}(\mathbf{R}^{2,3}_i - \mathbf{R}^{3,2}_i) \frac{1}{2}
            \sqrt{1 + \mathbf{R}^{1,1}_i - \mathbf{R}^{2,2}_i - \mathbf{R}^{3,3}_i}\\
        q^y_i &= \mathrm{sign}(\mathbf{R}^{3,1}_i - \mathbf{R}^{1,3}_i) \frac{1}{2}
            \sqrt{1 - \mathbf{R}^{1,1}_i + \mathbf{R}^{2,2}_i - \mathbf{R}^{3,3}_i}\\
        q^z_i &= \mathrm{sign}(\mathbf{R}^{1,2}_i - \mathbf{R}^{2,1}_i) \frac{1}{2}
            \sqrt{1 - \mathbf{R}^{1,1}_i - \mathbf{R}^{2,2}_i + \mathbf{R}^{3,3}_i}\\
        q^w_i &= \frac{1}{2} \sqrt{1 + \mathbf{R}^{1,1}_i + \mathbf{R}^{2,2}_i + \mathbf{R}^{3,3}_i}
        \end{aligned}\right.,

    In summary, the output LieTensor should be of format:

    .. math::
        \textbf{y}_i = [t^x_i, t^y_i, t^z_i, q^x_i, q^y_i, q^z_i, q^w_i]

    Warning:
        Numerically, a transformation matrix is considered legal if:

        .. math::
            |{\rm det}(\mathbf{R}) - 1| \leq \texttt{atol} + \texttt{rtol}\times 1\\
            |\mathbf{RR}^{T} - \mathbf{I}| \leq \texttt{atol} + \texttt{rtol}\times \mathbf{I}
        
        where :math:`|\cdot |` is element-wise absolute function. When ``check`` is set to ``True``,
        illegal input will raise a ``ValueError``. Otherwise, no data validation is performed. 
        Illegal input will output an irrelevant result, which likely contains ``nan``.

        For input with shape :obj:`(*, 4, 4)`, when ``check`` is set to ``True`` and the last row
        of the each individual matrix is not ``[0, 0, 0, 1]``, a warning will be triggered. 
        Even though the last row is not used in the computation, it is worth noting that a matrix not
        satisfying this condition is not a valid transformation matrix.

    Examples:

        >>> input = torch.tensor([[0., -1., 0., 0.1],
        ...                       [1.,  0., 0., 0.2],
        ...                       [0.,  0., 1., 0.3],
        ...                       [0.,  0., 0.,  1.]])
        >>> pp.mat2SE3(input)
        SE3Type LieTensor:
        tensor([0.1000, 0.2000, 0.3000, 0.0000, 0.0000, 0.7071, 0.7071])

    Note:
        The individual matrix in a batch can be written as:

        .. math::
            \begin{bmatrix}
                    \mathbf{R}_{3\times3} & \mathbf{t}_{3\times1}\\
                    \textbf{0} & 1
            \end{bmatrix},

        where :math:`\mathbf{R}` is the rotation matrix. The translation vector :math:`\mathbf{t}` defines the
        displacement between the original position and the transformed position.


    See :meth:`pypose.SE3` for more details of the output LieTensor format.
    """
    if not torch.is_tensor(mat):
        mat = torch.tensor(mat)

    if len(mat.shape) < 2:
        raise ValueError("Input size must be at least 2 dimensions. Got {}".format(mat.shape))

    if not (mat.shape[-2:] == (3, 3) or mat.shape[-2:] == (3, 4) or mat.shape[-2:] == (4, 4)):
        raise ValueError("Input size must be a * x 3 x 3 or * x 3 x 4 or * x 4 x 4  tensor. \
                Got {}".format(mat.shape))
    
    shape = mat.shape
    if shape[-2:] == (4, 4) and check == True:
        zerosone = torch.tensor([0, 0, 0, 1], dtype=mat.dtype, device=mat.device)
        if not torch.allclose(mat[..., 3, :], zerosone.expand_as(mat[..., 3, :]), rtol=rtol, atol=atol):
            warnings.warn("input of shape 4x4 last rows are not all equal [0, 0, 0, 1]")

    q = mat2SO3(mat[..., :3, :3], check=check, rtol=rtol, atol=atol).tensor()
    if shape[-1] == 3:
        t = torch.zeros(shape[:-2]+(3,), dtype=mat.dtype, device=mat.device, requires_grad=mat.requires_grad)
    else:
        t = mat[..., :3, 3]
    vec = torch.cat([t, q], dim=-1)

    return SE3(vec)


def mat2Sim3(mat, check=True, rtol=1e-5, atol=1e-5):
    r"""Convert batched rotation or transformation matrices to Sim3Type LieTensor.

    Args:
        mat (Tensor): the batched matrices to convert. If input is of shape :obj:`(*, 3, 3)`, 
            then translation will be filled with zero. For input with shape :obj:`(*, 3, 4)`, 
            the last row will be treated as ``[0, 0, 0, 1]``.
        check (bool, optional): flag to check if the input is valid rotation matrices (orthogonal
            and with a determinant of one). Set to ``False`` if less computation is needed.
            Default: ``True``.
        rtol (float, optional): relative tolerance when check is enabled. Default: 1e-05
        atol (float, optional): absolute tolerance when check is enabled. Default: 1e-05

    Return:
        LieTensor: the converted Sim3Type LieTensor.

    Shape:
        Input: :obj:`(*, 3, 3)` or :obj:`(*, 3, 4)` or :obj:`(*, 4, 4)`

        Output: :obj:`(*, 8)`

    Let the input be matrix :math:`\mathbf{T}`,  :math:`\mathbf{T}_i` represents each individual
    matrix in the batch. :math:`\mathbf{U}_i\in\mathbb{R}^{3\times 3}` be the top left 3x3 block 
    matrix of :math:`\mathbf{T}_i`. Let :math:`\mathbf{T}^{m,n}_i` represents the 
    :math:`m^{\mathrm{th}}` row and :math:`n^{\mathrm{th}}` column of :math:`\mathbf{T}_i`, 
    :math:`m,n\geq 1`, then the scaling factor :math:`s_i\in\mathbb{R}` and the rotation matrix
    :math:`\mathbf{R}_i\in\mathbb{R}^{3\times 3}` can be computed as:

    .. math::
        \begin{aligned}
            s_i &= \sqrt[3]{\vert \mathbf{U}_i \vert}\\
            \mathbf{R}_i &= \mathbf{U}_i/s_i
        \end{aligned}
        
    the translation and quaternion can be computed by:

    .. math::
        \left\{\begin{aligned}
        t^x_i &= \mathbf{T}^{1,4}_i\\
        t^y_i &= \mathbf{T}^{2,4}_i\\
        t^z_i &= \mathbf{T}^{3,4}_i\\
        q^x_i &= \mathrm{sign}(\mathbf{R}^{2,3}_i - \mathbf{R}^{3,2}_i) \frac{1}{2}
            \sqrt{1 + \mathbf{R}^{1,1}_i - \mathbf{R}^{2,2}_i - \mathbf{R}^{3,3}_i}\\
        q^y_i &= \mathrm{sign}(\mathbf{R}^{3,1}_i - \mathbf{R}^{1,3}_i) \frac{1}{2}
            \sqrt{1 - \mathbf{R}^{1,1}_i + \mathbf{R}^{2,2}_i - \mathbf{R}^{3,3}_i}\\
        q^z_i &= \mathrm{sign}(\mathbf{R}^{1,2}_i - \mathbf{R}^{2,1}_i) \frac{1}{2}
            \sqrt{1 - \mathbf{R}^{1,1}_i - \mathbf{R}^{2,2}_i + \mathbf{R}^{3,3}_i}\\
        q^w_i &= \frac{1}{2} \sqrt{1 + \mathbf{R}^{1,1}_i + \mathbf{R}^{2,2}_i + \mathbf{R}^{3,3}_i}
        \end{aligned}\right.,
    
    In summary, the output LieTensor should be of format:

    .. math::
        \textbf{y}_i = [t^x_i, t^y_i, t^z_i, q^x_i, q^y_i, q^z_i, q^w_i, s_i]

    Warning:
        Numerically, a transformation matrix is considered legal if:

        .. math::
            \vert s \vert > \texttt{atol} \\
            |{\rm det}(\mathbf{R}) - 1| \leq \texttt{atol} + \texttt{rtol}\times 1\\
            |\mathbf{RR}^{T} - \mathbf{I}| \leq \texttt{atol} + \texttt{rtol}\times \mathbf{I}
        
        where :math:`|\cdot |` is element-wise absolute function. When ``check`` is set to ``True``,
        illegal input will raise a ``ValueError``. Otherwise, no data validation is performed. 
        Illegal input will output an irrelevant result, which likely contains ``nan``.

        For input with shape :obj:`(*, 4, 4)`, when ``check`` is set to ``True`` and the last row
        of the each individual matrix is not ``[0, 0, 0, 1]``, a warning will be triggered. 
        Even though the last row is not used in the computation, it is worth noting that a matrix not
        satisfying this condition is not a valid transformation matrix.
        
    Examples:
        >>> input = torch.tensor([[ 0.,-0.5,  0., 0.1],
        ...                       [0.5,  0.,  0., 0.2],
        ...                       [ 0.,  0., 0.5, 0.3],
        ...                       [ 0.,  0.,  0.,  1.]])
        >>> pp.mat2Sim3(input)
        Sim3Type LieTensor:
        tensor([0.1000, 0.2000, 0.3000, 0.0000, 0.0000, 0.7071, 0.7071, 0.5000])

    Note:
        We follow the convention below to express Sim3:

        .. math::
            \begin{bmatrix}
                    s\mathbf{R}_{3\times3} & \mathbf{t}_{3\times1}\\
                    \textbf{0} & 1
            \end{bmatrix},

        referred to this paper:

        * J. Sola et al., `A micro Lie theory for state estimation in
          robotics <https://arxiv.org/abs/1812.01537>`_, arXiv preprint arXiv:1812.01537 (2018),
        
        where :math:`\mathbf{R}` is the individual matrix in a batch. The scaling factor 
        :math:`s` defines a linear transformation that enlarges or diminishes the object in the
        same ratio across 3 dimensions, the translation vector :math:`\mathbf{t}` defines the 
        displacement between the original position and the transformed position.

        We also notice that there is another popular convention:

        .. math::
            \begin{bmatrix}
                    \mathbf{R}_{3\times3} & \mathbf{t}_{3\times1}\\
                    \textbf{0} & 1/s
            \end{bmatrix},

        referred to this tutorial:

        * `Lie Groups for 2D and 3D Transformations.
          <https://www.ethaneade.org/lie.pdf>`_, by Ethan Eade.

        Please make sure your own convention before using this function.

    See :meth:`pypose.Sim3` for more details of the output LieTensor format.
    """
    if not torch.is_tensor(mat):
        mat = torch.tensor(mat)

    if len(mat.shape) < 2:
        raise ValueError("Input size must be at least 2 dimensions. Got {}".format(mat.shape))

    if not (mat.shape[-2:] == (3, 3) or mat.shape[-2:] == (3, 4) or mat.shape[-2:] == (4, 4)):
        raise ValueError("Input size must be a * x 3 x 3 or * x 3 x 4 or * x 4 x 4  tensor. \
                Got {}".format(mat.shape))
    
    shape = mat.shape
    if shape[-2:] == (4, 4) and check == True:
        zerosone = torch.tensor([0, 0, 0, 1], dtype=mat.dtype, device=mat.device)
        if not torch.allclose(mat[..., 3, :], zerosone.expand_as(mat[..., 3, :]), rtol=rtol, atol=atol):
            warnings.warn("Input of shape 4x4 last rows are not all equal [0, 0, 0, 1]")

    shape = mat.shape
    rot = mat[..., :3, :3]

    s = torch.pow(torch.det(mat), 1/3).unsqueeze(-1)
    zeros = torch.zeros(shape[:-2], dtype=mat.dtype, device=mat.device)
    if torch.allclose(s,  zeros, rtol=rtol, atol=atol):
        raise ValueError("Rotation matrix not full rank.")

    q = mat2SO3(rot/s.unsqueeze(-1), check=check, rtol=rtol, atol=atol).tensor()

    if mat.shape[-1] == 3:
        t = torch.zeros(mat.shape[:-2]+(3,), dtype=mat.dtype, device=mat.device, requires_grad=mat.requires_grad)
    else:
        t = mat[..., :3, 3]

    vec = torch.cat([t, q, s], dim=-1)

    return Sim3(vec)


def mat2RxSO3(mat, check=True, rtol=1e-5, atol=1e-5):
    r"""Convert batched rotation or transformation matrices to RxSO3Type LieTensor.

    Args:
        mat (Tensor): the batched matrices to convert. If input is of shape :obj:`(*, 3, 4)` 
            or :obj:`(*, 4, 4)`, only the top left 3x3 submatrix is used.
        check (bool, optional): flag to check if the input is valid rotation matrices (orthogonal
            and with a determinant of one). Set to ``False`` if less computation is needed.
            Default: ``True``.
        rtol (float, optional): relative tolerance when check is enabled. Default: 1e-05
        atol (float, optional): absolute tolerance when check is enabled. Default: 1e-05

    Return:
        LieTensor: the converted RxSO3Type LieTensor.

    Shape:
        Input: :obj:`(*, 3, 3)` or :obj:`(*, 3, 4)` or :obj:`(*, 4, 4)`

        Output: :obj:`(*, 5)`
    
    Let the input be matrix :math:`\mathbf{T}`, :math:`\mathbf{T}_i` represents each individual
    matrix in the batch. :math:`\mathbf{T}^{m,n}_i` represents the :math:`m^{\mathrm{th}}` row and
    :math:`n^{\mathrm{th}}` column of :math:`\mathbf{T}_i`, :math:`m,n\geq 1`, then the scaling factor 
    :math:`s_i\in\mathbb{R}` and the rotation matrix :math:`\mathbf{R}_i\in\mathbb{R}^{3\times 3}` 
    can be computed as:

    .. math::
        \begin{aligned}
            s_i &= \sqrt[3]{\vert \mathbf{T_i} \vert}\\
            \mathbf{R}_i &= \mathbf{R}_i/s_i
        \end{aligned},
    
    the translation and quaternion can be computed by:
    
    .. math::
        \left\{\begin{aligned}
        q^x_i &= \mathrm{sign}(\mathbf{R}^{2,3}_i - \mathbf{R}^{3,2}_i) \frac{1}{2}
            \sqrt{1 + \mathbf{R}^{1,1}_i - \mathbf{R}^{2,2}_i - \mathbf{R}^{3,3}_i}\\
        q^y_i &= \mathrm{sign}(\mathbf{R}^{3,1}_i - \mathbf{R}^{1,3}_i) \frac{1}{2}
            \sqrt{1 - \mathbf{R}^{1,1}_i + \mathbf{R}^{2,2}_i - \mathbf{R}^{3,3}_i}\\
        q^z_i &= \mathrm{sign}(\mathbf{R}^{1,2}_i - \mathbf{R}^{2,1}_i) \frac{1}{2}
            \sqrt{1 - \mathbf{R}^{1,1}_i - \mathbf{R}^{2,2}_i + \mathbf{R}^{3,3}_i}\\
        q^w_i &= \frac{1}{2} \sqrt{1 + \mathbf{R}^{1,1}_i + \mathbf{R}^{2,2}_i + \mathbf{R}^{3,3}_i}
        \end{aligned}\right.,
    
    In summary, the output LieTensor should be of format:

    .. math::
        \textbf{y}_i = [q^x_i, q^y_i, q^z_i, q^w_i, s_i]

    Warning:
        Numerically, a transformation matrix is considered legal if:

        .. math::
            \vert s \vert > \texttt{atol} \\
            |{\rm det}(\mathbf{R}) - 1| \leq \texttt{atol} + \texttt{rtol}\times 1\\
            |\mathbf{RR}^{T} - \mathbf{I}| \leq \texttt{atol} + \texttt{rtol}\times \mathbf{I}
        
        where :math:`|\cdot |` is element-wise absolute function. When ``check`` is set to ``True``,
        illegal input will raise a ``ValueError``. Otherwise, no data validation is performed. 
        Illegal input will output an irrelevant result, which likely contains ``nan``.

    Examples:
        >>> input = torch.tensor([[ 0., -0.5,  0.],
        ...                       [0.5,   0.,  0.],
        ...                       [ 0.,   0., 0.5]])
        >>> pp.mat2RxSO3(input)
        RxSO3Type LieTensor:
        tensor([0.0000, 0.0000, 0.7071, 0.7071, 0.5000])

    Note:
        The individual matrix in a batch can be written as: :math:`s\mathbf{R}_{3\times3}`, 
        where :math:`\mathbf{R}` is the rotation matrix. where  the scaling factor 
        :math:`s` defines a linear transformation that enlarges or diminishes the object 
        in the same ratio across 3 dimensions.

    See :meth:`pypose.RxSO3` for more details of the output LieTensor format.
    """
    if not torch.is_tensor(mat):
        mat = torch.tensor(mat)

    if len(mat.shape) < 2:
        raise ValueError("Input size must be at least 2 dimensions. Got {}".format(mat.shape))

    if not (mat.shape[-2:] == (3, 3) or mat.shape[-2:] == (3, 4) or mat.shape[-2:] == (4, 4)):
        raise ValueError("Input size must be a * x 3 x 3 or * x 3 x 4 or * x 4 x 4  tensor. \
                Got {}".format(mat.shape))

    shape = mat.shape
    rot = mat[..., :3, :3]

    s = torch.pow(torch.det(mat), 1/3).unsqueeze(-1)
    if torch.allclose(s,  torch.zeros(shape[:-2], dtype=mat.dtype, device=mat.device), rtol=rtol, atol=atol):
        raise ValueError("Rotation matrix not full rank.")

    q = mat2SO3(rot/s.unsqueeze(-1), check=check, rtol=rtol, atol=atol).tensor()
    vec = torch.cat([q, s], dim=-1)

    return RxSO3(vec)


def from_matrix(mat, ltype, check=True, rtol=1e-5, atol=1e-5):
    r"""Convert batched rotation or transformation matrices to LieTensor.

    Args:
        mat (Tensor): the matrix to convert.
        ltype (ltype): specify the LieTensor type, chosen from :class:`pypose.SO3_type`,
            :class:`pypose.SE3_type`, :class:`pypose.Sim3_type`, or :class:`pypose.RxSO3_type`.
            See more details in :meth:`LieTensor`
        check (bool, optional): flag to check if the input is valid rotation matrices (orthogonal
            and with a determinant of one). Set to ``False`` if less computation is needed.
            Default: ``True``.
        rtol (float, optional): relative tolerance when check is enabled. Default: 1e-05
        atol (float, optional): absolute tolerance when check is enabled. Default: 1e-05

    Warning:
        Numerically, a transformation matrix is considered legal if:

        .. math::
            |{\rm det}(\mathbf{R}) - 1| \leq \texttt{atol} + \texttt{rtol}\times 1\\
            |\mathbf{RR}^{T} - \mathbf{I}| \leq \texttt{atol} + \texttt{rtol}\times \mathbf{I}
        
        where :math:`|\cdot |` is element-wise absolute function. When ``check`` is set to ``True``,
        illegal input will raise a ``ValueError``. Otherwise, no data validation is performed. 
        Illegal input will output an irrelevant result, which likely contains ``nan``.
    
    Return:
        LieTensor: the converted LieTensor.
    Examples:

        - :class:`pypose.SO3_type`

        >>> pp.from_matrix(torch.tensor([[0., -1., 0.],
        ...                              [1.,  0., 0.],
        ...                              [0.,  0., 1.]]), ltype=pp.SO3_type)
        SO3Type LieTensor:
        tensor([0.0000, 0.0000, 0.7071, 0.7071])

        - :class:`pypose.SE3_type`

        >>> pp.from_matrix(torch.tensor([[0., -1., 0., 0.1],
        ...                              [1.,  0., 0., 0.2],
        ...                              [0.,  0., 1., 0.3],
        ...                              [0.,  0., 0.,  1.]]), ltype=pp.SE3_type)
        SE3Type LieTensor:
        tensor([0.1000, 0.2000, 0.3000, 0.0000, 0.0000, 0.7071, 0.7071])

        - :class:`pypose.Sim3_type`

        >>> pp.from_matrix(torch.tensor([[ 0.,-0.5,  0., 0.1],
        ...                              [0.5,  0.,  0., 0.2],
        ...                              [ 0.,  0., 0.5, 0.3],
        ...                              [ 0.,  0.,  0.,  1.]]), ltype=pp.Sim3_type)
        Sim3Type LieTensor:
        tensor([0.1000, 0.2000, 0.3000, 0.0000, 0.0000, 0.7071, 0.7071, 0.5000])

        - :class:`pypose.RxSO3_type`

        >>> pp.from_matrix(torch.tensor([[0., -0.5, 0.],
        ...                              [0.5, 0.,  0.],
        ...                              [0.,  0., 0.5]]), ltype=pp.RxSO3_type)
        RxSO3Type LieTensor:
        tensor([0.0000, 0.0000, 0.7071, 0.7071, 0.5000])
    """
    if not torch.is_tensor(mat):
        mat = torch.tensor(mat)

    if len(mat.shape) < 2:
        raise ValueError("Input size must be at least 2 dimensions. Got {}".format(mat.shape))

    if not (mat.shape[-2:] == (3, 3) or mat.shape[-2:] == (3, 4) or mat.shape[-2:] == (4, 4)):
        raise ValueError("Input size must be a * x 3 x 3 or * x 3 x 4 or * x 4 x 4  tensor. \
                Got {}".format(mat.shape))

    if ltype == SO3_type:
        return mat2SO3(mat, check=check, rtol=rtol, atol=atol)
    elif ltype == SE3_type:
        return mat2SE3(mat, check=check, rtol=rtol, atol=atol)
    elif ltype == Sim3_type:
        return mat2Sim3(mat, check=check, rtol=rtol, atol=atol)
    elif ltype == RxSO3_type:
        return mat2RxSO3(mat, check=check, rtol=rtol, atol=atol)
    else:
        raise ValueError("Input ltype must be one of SO3_type, SE3_type, Sim3_type or RxSO3_type.\
                Got {}".format(ltype))


def matrix(lietensor):
    assert isinstance(lietensor, LieTensor)
    return lietensor.matrix()


def euler2SO3(euler: torch.Tensor):
    r"""Convert batched Euler angles (roll, pitch, and yaw) to SO3Type LieTensor.

    Args:
        euler (Tensor): the euler angles in radians to convert.

    Return:
        LieTensor: the converted SO3Type LieTensor.

    Shape:
        Input: :obj:`(*, 3)`

        Output: :obj:`(*, 4)`

    .. math::
        {\displaystyle \mathbf{y}_i={
        \begin{bmatrix}\,
        \sin(\alpha_i)\cos(\beta_i)\cos(\gamma_i) - \cos(\alpha_i)\sin(\beta_i)\sin(\gamma_i)\\
        \cos(\alpha_i)\sin(\beta_i)\cos(\gamma_i) + \sin(\alpha_i)\cos(\beta_i)\sin(\gamma_i)\\
        \cos(\alpha_i)\cos(\beta_i)\sin(\gamma_i) - \sin(\alpha_i)\sin(\beta_i)\cos(\gamma_i)\\
        \cos(\alpha_i)\cos(\beta_i)\cos(\gamma_i) + \sin(\alpha_i)\sin(\beta_i)\sin(\gamma_i)
        \end{bmatrix}}},

    where the :math:`i`-th item of input :math:`\mathbf{x}_i = [\alpha_i, \beta_i, \gamma_i]`
    are roll, pitch, and yaw, respectively.

    Note:
        The last dimension of the input tensor has to be 3. The Euler angle takes the rotation
        sequence of x (roll), y (pitch), then z (yaw) axis (counterclockwise).

    Warning:
        Any given rotation has two possible quaternion representations. If one is known, the other
        is just the negative of all four terms. This function only returns one of them.

    Examples:
        >>> input = torch.randn(2, 3, requires_grad=True, dtype=torch.float64)
        >>> pp.euler2SO3(input)
        SO3Type LieTensor:
        tensor([[-0.4873,  0.1162,  0.4829,  0.7182],
                [ 0.3813,  0.4059, -0.2966,  0.7758]], dtype=torch.float64, grad_fn=<AliasBackward0>)

    See :obj:`euler` for more information.
    """
    if not torch.is_tensor(euler):
        euler = torch.tensor(euler)
    assert euler.shape[-1] == 3
    shape, euler = euler.shape, euler.view(-1, 3)
    roll, pitch, yaw = euler[:, 0], euler[:, 1], euler[:, 2]
    cy, sy = (yaw * 0.5).cos(), (yaw * 0.5).sin()
    cp, sp = (pitch * 0.5).cos(), (pitch * 0.5).sin()
    cr, sr = (roll * 0.5).cos(), (roll * 0.5).sin()

    q = torch.stack([sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy,
                     cr * cp * sy - sr * sp * cy,
                     cr * cp * cy + sr * sp * sy], dim=-1)
    return SO3(q).lview(*shape[:-1])


def tensor(inputs):
    r'''
    Convert a :obj:`LieTensor` into a :obj:`torch.Tensor` without changing data.

    Args:
        inputs (:obj:`LieTensor`): the input LieTensor.

    Return:
        Tensor: the torch.Tensor form of LieTensor.

    Example:
        >>> x = pp.randn_SO3(2)
        >>> x.tensor()
        tensor([[ 0.1196,  0.2339, -0.6824,  0.6822],
                [ 0.9198, -0.2704, -0.2395,  0.1532]])
    '''
    return inputs.tensor()


def translation(inputs):
    r'''
    Extract the translation part from a :obj:`LieTensor`.

    Args:
        inputs (:obj:`LieTensor`): the input LieTensor.

    Return:
        Tensor: the batched translation vectors.

    Warning:
        The :obj:`SO3`, :obj:`so3`, :obj:`RxSO3`, and :obj:`rxso3` types do not contain
        translation. Calling :obj:`translation()` on these types will return zero vector(s).

    Example:
        >>> x = pp.randn_SE3(2)
        >>> x.translation()
        tensor([[-0.5358, -1.5421, -0.7224],
                [ 0.8331, -1.4412,  0.0863]])
        >>> y = pp.randn_SO3(2)
        >>> y.translation()
        tensor([[0., 0., 0.],
                [0., 0., 0.]])
    '''
    return inputs.translation()


def rotation(inputs):
    r'''
    Extract the rotation part from a :obj:`LieTensor`.

    Args:
        inputs (:obj:`LieTensor`): the input LieTensor.

    Return:
        SO3: the batched quaternions.

    Example:
        >>> x = pp.randn_SE3(2)
        >>> x.rotation()
        SO3Type LieTensor:
        tensor([[-0.8302,  0.5200, -0.0056,  0.2006],
                [-0.2541, -0.3184,  0.6305,  0.6607]])
    '''
    return inputs.rotation()


def scale(inputs):
    r'''
    Extract the scale part from a :obj:`LieTensor`.

    Args:
        inputs (:obj:`LieTensor`): the input LieTensor.

    Return:
        Tensor: the batched scale scalars.

    Warning:
        The :obj:`SO3`, :obj:`so3`, :obj:`SE3`, and :obj:`se3` types do not contain scale. 
        Calling :obj:`scale()` on these types will return one(s).

    Example:
        >>> x = pp.randn_Sim3(4)
        >>> x.scale()
        tensor([[10.9577],
                [ 1.0248],
                [ 0.0947],
                [ 1.1989]])
        >>> y = pp.randn_SE3(2)
        >>> y.scale()
        tensor([[1.],
                [1.]])
        '''
    return inputs.scale()


def matrix(inputs):
    r'''
    Convert a :obj:`LieTensor` into matrix form.

    Args:
        inputs (:obj:`LieTensor`): the input LieTensor.

    Return:
        Tensor: the batched matrix form (torch.Tensor) of LieTensor.

    Example:
        >>> x = pp.randn_SO3(2)
        >>> x.matrix()
        tensor([[[ 0.9285, -0.0040, -0.3713],
                 [ 0.2503,  0.7454,  0.6178],
                 [ 0.2743, -0.6666,  0.6931]],
                [[ 0.4805,  0.8602, -0.1706],
                 [-0.7465,  0.2991, -0.5944],
                 [-0.4603,  0.4130,  0.7858]]])
    '''
    return inputs.matrix()


def euler(inputs, eps=2e-4):
    r'''
    Convert batched LieTensor into Euler angles (roll, pitch, yaw).

    Args:
        inputs (:obj:`LieTensor`): the input LieTensor.
        eps (:obj:`float`, optional): the threshold to avoid the sigularity caused 
            by gimbal lock. Default: 2e-4.

    Return:
        :obj:`Tensor`: the batched Euler angles in radians.

    Supported input type: :obj:`so3`, :obj:`SO3`, :obj:`se3`, :obj:`SE3`,
    :obj:`sim3`, :obj:`Sim3`, :obj:`rxso3`, and :obj:`RxSO3`.

    Warning:
        The Euler angle takes the rotation sequence of x (roll), y (pitch),
        and z (yaw) axis (counterclockwise). There is always more than one
        solution that can result in the same orientation, while this function
        only returns one of them.

        When the pitch angle is around :math:`\pm \frac{\pi}{2}` (north/south pole), 
        there will be sigularity problem due to the `gimbal lock
        <https://en.wikipedia.org/wiki/Gimbal_lock>`_ and some information can be
        found in `this pape <https://tinyurl.com/py8frs6v>`_.

    Example:

        >>> x = pp.randn_SO3()
        >>> x.euler()  # equivalent to pp.euler(x)
        tensor([-0.6599,  0.2749, -0.3263])

        >>> x = pp.randn_Sim3(2)
        >>> x.euler()  # equivalent to pp.euler(x)
        tensor([[-0.2701, -0.8006, -0.4150],
                [ 2.1550,  0.1768,  0.9368]])

        >>> x = pp.randn_rxso3()
        >>> x.euler()  # equivalent to pp.euler(x)
        tensor([ 1.2676, -0.4783, -0.3596])

    See :obj:`euler2SO3` for more information.
    '''
    return inputs.euler(eps=eps)
