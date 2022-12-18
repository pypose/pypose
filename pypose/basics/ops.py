import torch


def bmv(input, vec, *, out=None):
    r'''
    Performs batched matrix-vector product.

    Args:
        input (:obj:`Tensor`): matrices to be multiplied.
        vec (:obj:`Tensor`): vectors to be multiplied.
    
    Return:
        out (:obj:`Tensor`): the output tensor.

    Note:
        The ``input`` has to be a (:math:`\cdots\times n \times m`) tensor,
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
    '''
    assert input.ndim >= 2 and vec.ndim >= 1, 'Input arguments invalid'
    assert input.shape[-1] == vec.shape[-1], 'matrix-vector shape invalid'
    return torch.matmul(input, vec.unsqueeze(-1), out=out).squeeze(-1)
