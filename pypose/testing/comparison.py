import torch
from ..function.checking import is_lietensor


def assert_close(actual, expected, *args, **kwargs):
    '''
    Asserts that ``actual`` and ``expected`` are close. This function is exactly the same
    with `torch.testing.assert_close <https://tinyurl.com/3bm33ps7>`_ except for that it
    also accepts ``pypose.LieTensor``.

    Args:
        actual(:obj:`Tensor` or :obj:`LieTensor`): Actual input.
        expected(:obj:`Tensor` or :obj:`LieTensor`): Expected input.
        rtol (Optional[float]): Relative tolerance. If specified ``atol`` must also be
            specified. If omitted, default values based on the :attr:`~torch.Tensor.dtype`
            are selected with the below table.
        atol (Optional[float]): Absolute tolerance. If specified ``rtol`` must also be
            specified. If omitted, default values based on the :attr:`~torch.Tensor.dtype`
            are selected with the below table.

    If :math:`T_e` and :math:`T_a` are Lietensor, they are considered close if
    :math:`T_e*T_a^{-1} = \mathbf{O}`, where :math:`\mathbf{O}` is close to zero tensor in
    the sense of ``torch.testing.assert_close`` is ``True.``

    Warning:
        The prerequisites for the other arguments align precisely with
        `torch.testing.assert_close <https://tinyurl.com/3bm33ps7>`_. Kindly consult it
        for further details.

    Examples:
        >>> import pypose as pp
        >>> actual = pp.randn_SE3(3)
        >>> expected = actual.Log().Exp()
        >>> pp.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    '''
    if is_lietensor(actual) and is_lietensor(expected):
        source = (actual.Inv() @ expected).Log().tensor()
        target = torch.zeros_like(source)
    else:
        source, target = actual, expected
    torch.testing.assert_close(source, target, *args, **kwargs)
