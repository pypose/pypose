import torch
from .utils import SO3, so3
from .lietensor import LieTensor


def mat2SO3(rotation_matrix):
    r"""Convert batched 3x3 or 3x4 rotation matrices to SO3Type LieTensor.

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        LieTensor: the converted SO3Type LieTensor.

    Shape:
        Input: :obj:`(*, 3, 3)` or :obj:`(*, 3, 4)`

        Output: :obj:`(*, 4)`

    Examples:

        >>> input = torch.eye(3).repeat(2, 1, 1) # N x 3 x 3
        >>> pp.mat2SO3(input)                    # N x 4
        SO3Type LieTensor:
        tensor([[0., 0., 0., 1.],
                [0., 0., 0., 1.]])
    """
    if not torch.is_tensor(rotation_matrix):
        rotation_matrix = torch.tensor(rotation_matrix)

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))
    if not (rotation_matrix.shape[-2:] == (3, 3) or rotation_matrix.shape[-2:] == (3, 4)):
        raise ValueError(
            "Input size must be a * x 3 x 4 or * x 3 x 3  tensor. Got {}".format(
                rotation_matrix.shape))
    shape = rotation_matrix.shape
    rotation_matrix = rotation_matrix.view(-1, shape[-2], shape[-1])
    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < 1e-6

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= 2*torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa

    q = q.view(shape[:-2]+(4,))
    q = q.index_select(-1, torch.tensor([1,2,3,0], device=q.device)) # wxyz -> xyzw

    return SO3(q)


def euler2SO3(euler:torch.Tensor):
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
    roll, pitch, yaw = euler[:,0], euler[:,1], euler[:,2]
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
        inputs (LieTensor): the input LieTensor.

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

    Return:
        Tensor: the batched translation vectors.

    Warning:
        The :obj:`SO3`, :obj:`so3`, :obj:`RxSO3`, and :obj:`rxso3` types do not contain translation. 
        Calling :obj:`translation()` on these types will return zero vector(s).

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
