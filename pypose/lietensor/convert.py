import warnings
import torch

from pypose.lietensor.lietensor import LieTensor, SE3_type, SO3_type, Sim3_type, RxSO3_type
from .utils import SO3, SE3, RxSO3, Sim3


def mat2SO3(mat):
    r"""Convert batched 3x3 or 3x4 rotation matrices to SO3Type LieTensor.

    Args:
        mat (Tensor): the rotation matrix to convert.

    Return:
        LieTensor: the converted SO3Type LieTensor.

    Shape:
        Input: :obj:`(*, 3, 3)` or :obj:`(*, 3, 4)`

        Output: :obj:`(*, 4)`

    Warning:
        Illegal input(not full rank or not orthogonal) triggers a warning, but will output a quaternion regardless. 

    Suppose the input rotation matrix :math:`\mathbf{R}_i` is

    .. math::
        \mathbf{R}_i = \begin{bmatrix}
            R_{11} & R_{12} & R_{13} \\
            R_{21} & R_{22} & R_{23} \\
            R_{31} & R_{32} & R_{33}
        \end{bmatrix},

    the corresponding quaternion :math:`\mathbf{q}_i=\begin{bmatrix} q_x & q_y & q_z & q_w \end{bmatrix}` can be calculated by

    .. math::
        \left\{\begin{aligned}
        q_w &= \frac{1}{2} \sqrt{1 + R_{11} + R_{22} + R_{33}} \\
        q_x &= \mathrm{sign}(R_{23} - R_{32}) \frac{1}{2} \sqrt{1 + R_{11} - R_{22} - R_{33}} \\
        q_y &= \mathrm{sign}(R_{31} - R_{13}) \frac{1}{2} \sqrt{1 - R_{11} + R_{22} - R_{33}} \\
        q_z &= \mathrm{sign}(R_{12} - R_{21}) \frac{1}{2} \sqrt{1 - R_{11} - R_{22} + R_{33}}
        \end{aligned}\right..

    Examples:

        >>> input = torch.eye(3).repeat(2, 1, 1) # N x 3 x 3
        >>> pp.mat2SO3(input)                    # N x 4
        SO3Type LieTensor:
        tensor([[0., 0., 0., 1.],
                [0., 0., 0., 1.]])
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

    if not torch.all(torch.det(mat) > 0):
        warnings.warn(
            "Input rotation matrices are not all full rank", RuntimeWarning)

    mat = mat[..., :3, :3]
    shape = mat.shape
    if not (torch.allclose(torch.sum(mat[..., 0]*mat[..., 1], dim=-1), torch.zeros(shape[0]), atol=1e-6)
            and torch.allclose(torch.sum(mat[..., 0]*mat[..., 2], dim=-1), torch.zeros(shape[0]), atol=1e-6)
            and torch.allclose(torch.sum(mat[..., 1]*mat[..., 2], dim=-1), torch.zeros(shape[0]), atol=1e-6)):
        warnings.warn(
            "Input rotation matrices are not all orthogonal matrix", RuntimeWarning)

    rmat_t = torch.transpose(mat, -1, -2)

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
    r"""Convert batched 3x3 or 3x4 tranformation matrices to SO3Type LieTensor.

    Args:
        mat (Tensor): the tranformation matrix to convert.

    Return:
        LieTensor: the converted SE3Type LieTensor.

    Shape:
        Input: :obj:`(*, 3, 4)` or :obj:`(*, 4, 4)`

        Output: :obj:`(*, 7)`

    Examples:

        >>> input = torch.eye(4).repeat(2, 1, 1) # N x 3 x 3
        >>> pp.mat2SE3(input)                    # N x 7
        SE3Type LieTensor:
        tensor([[0., 0., 0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 0., 1.]])
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

    q = mat2SO3(mat[..., :3, :3]).tensor()
    if mat.shape[-1] == 3:
        t = torch.zeros(mat.shape[:-2]+torch.Size((3,)))
    else:
        t = mat[..., :3, 3]
    print(t.shape, q.shape)
    vec = torch.cat([t, q], dim=-1)

    return SE3(vec)


def mat2Sim3(mat):
    r"""Convert batched 3x4 or 4x4 tranformation matrices to Sim3Type LieTensor.

    Args:
        mat (Tensor): the tranformation matrix to convert.

    Return:
        LieTensor: the converted Sim3Type LieTensor.

    Shape:
        Input: :obj:`(*, 3, 4)` or :obj:`(*, 4, 4)`

        Output: :obj:`(*, 8)`

    Examples:

        >>> input = torch.eye(4).repeat(2, 1, 1) # N x 3 x 3
        >>> pp.mat2Sim3(input)                   # N x 8
        Sim3Type LieTensor:
        tensor([[0., 0., 0., 0., 0., 0., 1., 1.],
                [0., 0., 0., 0., 0., 0., 1., 1.]])
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
        t = torch.zeros(mat.shape[:-2]+torch.Size((3,)))
    else:
        t = mat[..., :3, 3]

    vec = torch.cat([t, q, s], dim=-1)

    return Sim3(vec)


def mat2RxSO3(mat):
    r"""Convert batched 3x3 or 3x4 rotation matrices to RxSO3Type LieTensor.

    Args:
        mat (Tensor): the rotation matrix to convert.

    Return:
        LieTensor: the converted RxSO3Type LieTensor.

    Shape:
        Input: :obj:`(*, 3, 3)` or :obj:`(*, 3, 4)`

        Output: :obj:`(*, 5)`

    Examples:

        >>> input = torch.eye(3).repeat(2, 1, 1) # N x 3 x 3
        >>> pp.mat2RxSO3(input)                    # N x 4
            RxSO3Type LieTensor:
            tensor([[0., 0., 0., 1., 1.],
                    [0., 0., 0., 1., 1.]])
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
    r"""Convert batched 3x3 or 3x4 or 4x4 transformation matrices to LieTensor. The `ltype` will be automatically determined by the data.

    Args:
        mat (Tensor): the transformation matrix to convert.

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