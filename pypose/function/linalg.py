import torch
from .. import LieTensor


def bvv(lvec, rvec, *, out=None):
    r"""
    Performs batched vector-vector product, which results in matrices.

    Args:
        lvec (:obj:`Tensor`): left vectors to be multiplied.
        rvec (:obj:`Tensor`): right vectors to be multiplied.

    Return:
        out (:obj:`Tensor`): the output tensor.

    Note:
        This function is broadcastable and supports batched product.

    Example:
        >>> lvec = torch.randn(2, 1, 3)
        >>> rvec = torch.randn(1, 2, 2)
        >>> out = pp.bvv(lvec, rvec)
        >>> out.shape
        torch.Size([2, 2, 3, 2])
    """
    lvec = lvec.tensor() if isinstance(lvec, LieTensor) else lvec
    rvec = rvec.tensor() if isinstance(rvec, LieTensor) else rvec
    lvec, rvec = lvec.unsqueeze(-1), rvec.unsqueeze(-1)
    return torch.matmul(lvec, rvec.mT, out=out)


def bmv(mat, vec, *, out=None):
    r"""
    Performs batched matrix-vector product.

    Args:
        mat (:obj:`Tensor`): matrices to be multiplied.
        vec (:obj:`Tensor`): vectors to be multiplied.

    Return:
        out (:obj:`Tensor`): the output tensor.

    Note:
        The ``mat`` has to be a (:math:`\cdots\times n \times m`) tensor,
        the ``vec`` has to be a (:math:`\cdots\times m`) tensor,
        and ``out`` will be a (:math:`\cdots\times n`) tensor.
        Different from ``torch.mv``, which is not broadcast, this function
        is broadcast and supports batched product.

    Example:
        >>> matrix = torch.randn(2, 1, 3, 2)
        >>> vec = torch.randn(1, 2, 2)
        >>> out = pp.bmv(matrix, vec)
        >>> out.shape
        torch.Size([2, 2, 3])
    """
    assert mat.ndim >= 2 and vec.ndim >= 1, "Input arguments invalid"
    assert mat.shape[-1] == vec.shape[-1], "matrix-vector shape invalid"
    # mat = mat.tensor() if isinstance(mat, LieTensor) else mat
    # vec = vec.tensor() if isinstance(vec, LieTensor) else vec
    mat_clone = mat.clone()
    vec_clone = vec.clone()
    mat_clone = mat_clone.tensor() if isinstance(mat, LieTensor) else mat_clone
    vec_clone = vec_clone.tensor() if isinstance(vec, LieTensor) else vec_clone

    return torch.matmul(mat_clone, vec_clone.unsqueeze(-1), out=out).squeeze(-1)


def bvmv(lvec, mat, rvec):
    r"""
    Performs batched vector-matrix-vector product.

    .. math::
        \text{out}_i = \mathbf{v}_i^{l\text{T}} \times \mathbf{M}_i \times \mathbf{v}_i^r

    where :math:`\text{out}_i` is a scalar and :math:`\mathbf{v}_i^l, \mathbf{M}_i,
    \mathbf{v}_i^r` are the i-th batched tensors with shape (n), (n, m), and (m),
    respectively.

    Args:
        lvec (:obj:`Tensor`): left vectors to be multiplied.
        mat (:obj:`Tensor`): matrices to be multiplied.
        rvec (:obj:`Tensor`): right vectors to be multiplied.

    Return:
        :obj:`Tensor`: the output tensor.

    Note:
        the ``lvec`` has to be a (:math:`\cdots\times n`) tensor,
        The ``mat`` has to be a (:math:`\cdots\times n \times m`) tensor,
        the ``rvec`` has to be a (:math:`\cdots\times m`) tensor,
        and ``out`` will be a (:math:`\cdots`) or at least a 1D tensor.

    Example:
        >>> v1 = torch.randn(4)
        >>> mat = torch.randn(4, 5)
        >>> v2 = torch.randn(5)
        >>> out = pp.bvmv(v1, mat, v2)
        >>> out.shape
        torch.Size([1])

        >>> v1 = torch.randn(1, 2, 4)
        >>> mat = torch.randn(2, 2, 4, 5)
        >>> v2 = torch.randn(2, 1, 5)
        >>> out = pp.bvmv(v1, mat, v2)
        >>> out.shape
        torch.Size([2, 2])
    """
    assert mat.ndim >= 2 and lvec.ndim >= 1 and rvec.ndim >= 1, "Shape invalid"
    assert lvec.shape[-1] == mat.shape[-2] and mat.shape[-1] == rvec.shape[-1]
    lvec_clone = lvec.clone()
    mat_clone = mat.clone()
    rvec_clone = rvec.clone()
    lvec_clone = (
        lvec_clone.tensor() if isinstance(lvec_clone, LieTensor) else lvec_clone
    )
    mat_clone = mat_clone.tensor() if isinstance(mat_clone, LieTensor) else mat_clone
    rvec_clone = (
        rvec_clone.tensor() if isinstance(rvec_clone, LieTensor) else rvec_clone
    )

    lvec_clone, rvec_clone = lvec_clone.unsqueeze(-1), rvec_clone.unsqueeze(-1)
    return torch.atleast_1d(
        (lvec_clone.mT @ mat_clone @ rvec_clone).squeeze(-1).squeeze(-1)
    )
