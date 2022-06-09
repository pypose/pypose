import warnings
import torch

from pypose.lietensor.lietensor import LieTensor, SE3_type, SO3_type, Sim3_type, RxSO3_type
from .utils import SO3, SE3, RxSO3, Sim3


def mat2SO3(mat, check=False):
    r"""Convert batched rotation or transformation matrices to SO3Type LieTensor.

    Args:
        mat (Tensor): the matrix to convert.
        check (bool, optional): flag to check if the input is valid rotation matrices (orthogonal
            and and with a determinant of one). More computation is needed if ``True``.
            Default: ``False``.

    Return:
        LieTensor: the converted SO3Type LieTensor.

    Shape:
        Input: :obj:`(*, 3, 3)` or :obj:`(*, 3, 4)` or :obj:`(*, 4, 4)`

        Output: :obj:`(*, 4)`

    .. math::
        \mathbf{q}_i = 
        \left\{\begin{aligned}
        &\frac{1}{2} \sqrt{1 + R^{11}_i + R^{22}_i + R^{33}_i} \\
        &\mathrm{sign}(R^{23}_i - R^{32}_i) \frac{1}{2} \sqrt{1 + R^{11}_i - R^{22}_i - R^{33}_i}\\
        &\mathrm{sign}(R^{31}_i - R^{13}_i) \frac{1}{2} \sqrt{1 - R^{11}_i + R^{22}_i - R^{33}_i}\\
        &\mathrm{sign}(R^{12}_i - R^{21}_i) \frac{1}{2} \sqrt{1 - R^{11}_i - R^{22}_i + R^{33}_i}
        \end{aligned}\right.,
    
    where :math:`R` and :math:`\mathbf{q}=[]` are the input matrices and output LieTensor, respectively.

    Warning:
        Illegal input (not full rank or not orthogonal) triggers a warning, but will output a
        quaternion regardless.

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

    if check:
        e0 = mat @ mat.mT
        e1 = torch.eye(3, dtype=mat.dtype).repeat(shape[:-2] + (1, 1))
        if not torch.allclose(e0, e1, atol=torch.finfo(e0.dtype).resolution):
            warnings.warn("Input rotation matrices are not all orthogonal matrix, \
                the result is likely to be wrong", RuntimeWarning)

        if not torch.allclose(torch.det(mat), torch.ones(shape[:-2], dtype=mat.dtype)):
            warnings.warn("Input rotation matrices' determinant are not all equal to 1, \
                the result is likely to be wrong", RuntimeWarning)

    rmat_t = mat.mT

    mask_d2 = rmat_t[..., 2, 2] < 1e-6

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


def mat2SE3(mat):
    r"""Convert batched 3x3 or 3x4 or 4x4 matrices to SO3Type LieTensor.

    Args:
        mat (Tensor): the matrix to convert. If input is of shape :obj:`(*, 3, 3)`, then translation
            will be filled with zero.

    Return:
        LieTensor: the converted SE3Type LieTensor.

    Shape:
        Input: :obj:`(*, 3, 3)` or :obj:`(*, 3, 4)` or :obj:`(*, 4, 4)`

        Output: :obj:`(*, 7)`

    Suppose the input transformation matrix :math:`\mathbf{T}_i` is

    .. math::
        \mathbf{T}_i = \begin{bmatrix}
            R_{11} & R_{12} & R_{13} & t_x\\
            R_{21} & R_{22} & R_{23} & t_y\\
            R_{31} & R_{32} & R_{33} & t_z\\
            0 & 0 & 0 & 1
        \end{bmatrix},

    Output LieTensor should be of format:

    .. math::
        \mathrm{vec}[*, :] = [t_x, t_y, t_z, q_x, q_y, q_z, q_w]

    where :math:`\begin{pmatrix} q_x & q_y & q_z & q_w \end{pmatrix}^T` is computed by :meth:`pypose.mat2SO3`.

    Examples:

        >>> input = torch.tensor([[0., -1., 0., 0.1],
        ...                       [1., 0., 0., 0.2],
        ...                       [0., 0., 1., 0.3],
        ...                       [0., 0., 0., 1.]])
        >>> pp.mat2SE3(input)
        SE3Type LieTensor:
        tensor([0.1000, 0.2000, 0.3000, 0.0000, 0.0000, 0.7071, 0.7071])

    See :meth:`pypose.SE3` for more details of the output LieTensor format.
    """
    if not torch.is_tensor(mat):
        mat = torch.tensor(mat)

    if len(mat.shape) < 2:
        raise ValueError("Input size must be at least 2 dimensions. Got {}".format(mat.shape))

    if not (mat.shape[-2:] == (3, 3) or mat.shape[-2:] == (3, 4) or mat.shape[-2:] == (4, 4)):
        raise ValueError("Input size must be a * x 3 x 3 or * x 3 x 4 or * x 4 x 4  tensor. \
                            Got {}".format(mat.shape))

    q = mat2SO3(mat[..., :3, :3]).tensor()
    if mat.shape[-1] == 3:
        t = torch.zeros(mat.shape[:-2]+(3,))
    else:
        t = mat[..., :3, 3]
    vec = torch.cat([t, q], dim=-1)

    return SE3(vec)


def mat2Sim3(mat):
    r"""Convert batched 3x3 or 3x4 or 4x4 matrices to Sim3Type LieTensor.

    Args:
        mat (Tensor): the matrix to convert. If input is of shape :obj:`(*, 3, 3)`, then translation will be filled with zero.

    Return:
        LieTensor: the converted Sim3Type LieTensor.

    Shape:
        Input: :obj:`(*, 3, 3)` or :obj:`(*, 3, 4)` or :obj:`(*, 4, 4)`

        Output: :obj:`(*, 8)`

    Suppose the input transformation matrix :math:`\mathbf{T}_i` is

    .. math::
        \mathbf{T}_i = \begin{bmatrix}
            sR_{11} & sR_{12} & sR_{13} & t_x\\
            sR_{21} & sR_{22} & sR_{23} & t_y\\
            sR_{31} & sR_{32} & sR_{33} & t_z\\
            0 & 0 & 0 & 1
        \end{bmatrix},


    Output LieTensor should be of format:

    .. math::
        \mathrm{vec}[*, :] = [t_x, t_y, t_z, q_x, q_y, q_z, q_w, s]

    where :math:`\begin{pmatrix} q_x & q_y & q_z & q_w \end{pmatrix}^T` is computed by :meth:`pypose.mat2SO3`.

    Examples:

        >>> input = torch.tensor([[0., -0.5,  0., 0.1],
        ...                       [0.5,  0.,  0., 0.2],
        ...                       [ 0.,  0., 0.5, 0.3],
        ...                       [ 0.,  0.,  0.,  1.]])
        >>> pp.mat2Sim3(input)
        Sim3Type LieTensor:
        tensor([0.1000, 0.2000, 0.3000, 0.0000, 0.0000, 0.7071, 0.7071, 0.5000])

    See :meth:`pypose.Sim3` for more details of the output LieTensor format.
    """
    if not torch.is_tensor(mat):
        mat = torch.tensor(mat)

    if len(mat.shape) < 2:
        raise ValueError(
            "Input size must be at least 2 dimensions. Got {}".format(
                mat.shape))
    if not (mat.shape[-2:] == (3, 3) or mat.shape[-2:] == (3, 4) or mat.shape[-2:] == (4, 4)):
        raise ValueError(
            "Input size must be a * x 3 x 3 or * x 3 x 4 or * x 4 x 4  tensor. Got {}".format(
                mat.shape))

    rot = mat[..., :3, :3]

    s = torch.linalg.norm(rot[..., 0], dim=-1).unsqueeze(-1)
    q = mat2SO3(rot/s.unsqueeze(-1)).tensor()
    if mat.shape[-1] == 3:
        t = torch.zeros(mat.shape[:-2]+(3,))
    else:
        t = mat[..., :3, 3]

    vec = torch.cat([t, q, s], dim=-1)

    return Sim3(vec)


def mat2RxSO3(mat):
    r"""Convert batched 3x3 or 3x4 or 4x4 matrices to RxSO3Type LieTensor.

    Args:
        mat (Tensor): the matrix to convert.

    Return:
        LieTensor: the converted RxSO3Type LieTensor.

    Shape:
        Input: :obj:`(*, 3, 3)` or :obj:`(*, 3, 4)` or :obj:`(*, 4, 4)`

        Output: :obj:`(*, 5)`

    Suppose the input transformation matrix :math:`\mathbf{R}_i` is

    .. math::
        \mathbf{R}_i = \begin{bmatrix}
            sR_{11} & sR_{12} & sR_{13}\\
            sR_{21} & sR_{22} & sR_{23}\\
            sR_{31} & sR_{32} & sR_{33}
        \end{bmatrix},


    Output LieTensor should be of format:

    .. math::
        \mathrm{vec}[*, :] = [q_x, q_y, q_z, q_w, s]

    where :math:`\begin{pmatrix} q_x & q_y & q_z & q_w \end{pmatrix}^T` is computed by :meth:`pypose.mat2SO3`.

    Examples:

        >>> input = torch.tensor([[0., -0.5, 0.],
        ...                       [0.5, 0., 0.],
        ...                       [0., 0., 0.5]])
        >>> pp.mat2RxSO3(input)
        RxSO3Type LieTensor:
        tensor([0.0000, 0.0000, 0.7071, 0.7071, 0.5000])

    See :meth:`pypose.RxSO3` for more details of the output LieTensor format.
    """
    if not torch.is_tensor(mat):
        mat = torch.tensor(mat)

    if len(mat.shape) < 2:
        raise ValueError(
            "Input size must be at least 2 dimensions. Got {}".format(
                mat.shape))
    if not (mat.shape[-2:] == (3, 3) or mat.shape[-2:] == (3, 4) or mat.shape[-2:] == (4, 4)):
        raise ValueError(
            "Input size must be a * x 3 x 3 or * x 3 x 4 or * x 4 x 4  tensor. Got {}".format(
                mat.shape))

    rot = mat[..., :3, :3]

    s = torch.linalg.norm(rot[..., 0], dim=-1).unsqueeze(-1)
    q = mat2SO3(rot/s.unsqueeze(-1)).tensor()
    vec = torch.cat([q, s], dim=-1)

    return RxSO3(vec)


def from_matrix(mat, ltype):
    r"""Convert batched 3x3 or 3x4 or 4x4 matrices to LieTensor.

    Args:
        mat (Tensor): the matrix to convert.
        ltype (class): one of :meth:`pypose.SO3`, :meth:`pypose.SE3`, :meth:`pypose.Sim3` or :meth:`pypose.RxSO3`.

    Return:
        LieTensor: the converted LieTensor.
    """
    if not torch.is_tensor(mat):
        mat = torch.tensor(mat)

    if len(mat.shape) < 2:
        raise ValueError(
            "Input size must be at least 2 dimensions. Got {}".format(
                mat.shape))
    if not (mat.shape[-2:] == (3, 3) or mat.shape[-2:] == (3, 4) or mat.shape[-2:] == (4, 4)):
        raise ValueError(
            "Input size must be a * x 3 x 3 or * x 3 x 4 or * x 4 x 4  tensor. Got {}".format(
                mat.shape))

    if ltype == SO3_type:
        return mat2SO3(mat)
    elif ltype == SE3_type:
        return mat2SE3(mat)
    elif ltype == Sim3_type:
        return mat2Sim3(mat)
    elif ltype == RxSO3_type:
        return mat2RxSO3(mat)
    else:
        raise ValueError(
            "Input ltype must be one of SO3_type, SE3_type, Sim3_type or RxSO3_type. Got {}".format(
                ltype)
        )


def matrix(lietensor):
    assert isinstance(lietensor, LieTensor)
    return lietensor.matrix()


def euler2SO3(euler: torch.Tensor):
    r"""Convert batched Euler angles (roll, pitch, and yaw) to SO3Type LieTensor.

    Args:
        euler (Tensor): the euler angles to convert.

    Return:
        LieTensor: the converted SO3Type LieTensor.

    Shape:
        Input: :obj:`(*, 3)`

        Output: :obj:`(*, 4)`

    .. math::
        {\displaystyle \mathbf{y}_i={\begin{bmatrix}\,
        \sin(\alpha_i)\cos(\beta_i)\cos(\gamma_i) - \cos(\alpha_i)\sin(\beta_i)\sin(\gamma_i)\\\,
        \cos(\alpha_i)\sin(\beta_i)\cos(\gamma_i) + \sin(\alpha_i)\cos(\beta_i)\sin(\gamma_i)\\\,
        \cos(\alpha_i)\cos(\beta_i)\sin(\gamma_i) - \sin(\alpha_i)\sin(\beta_i)\cos(\gamma_i)\\\,
        \cos(\alpha_i)\cos(\beta_i)\cos(\gamma_i) + \sin(\alpha_i)\sin(\beta_i)\sin(\gamma_i)\end{bmatrix}}},

    where the :math:`i`-th item of input :math:`\mathbf{x}_i = [\alpha_i, \beta_i, \gamma_i]`
    are roll, pitch, and yaw, respectively.

    Note:
        The last dimension of the input tensor has to be 3.

    Examples:
        >>> input = torch.randn(2, 3, requires_grad=True, dtype=torch.float64)
        >>> pp.euler2SO3(input)
        SO3Type LieTensor:
        tensor([[-0.4873,  0.1162,  0.4829,  0.7182],
                [ 0.3813,  0.4059, -0.2966,  0.7758]], dtype=torch.float64, grad_fn=<AliasBackward0>)
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
