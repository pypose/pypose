import torch

def assert_lietensor_close(actual, expected, *args, **kwargs):
    '''
    Asserts that ``actual`` and ``expected`` are close.

    Args:
        actual(:obj:`LieTensor`): Actual Lietensor input.
        expected(:obj:`LieTensor`): Expected Lietensor input.
        *args: parameters in torch.testing.assert_close
        **kwargs: parameters in torch.testing.assert_close

    If :math:`T_e` and :math:`T_a` are the expected Lietensor and actual Lietensor, they
    are considered close if :math:`T_e*T_a^{-1} = \mathbf{O}`, where :math:`\mathbf{O}`
    is a zero matrix.

    Note:
        The details of parameters ``*args`` and ``**kwargs`` can be found in
        `torch.testing.assert_close <https://pytorch.org/docs/stable/testing.html>`_.

    Examples:
        >>> import pypose as pp
        >>> actual = pp.randn_SE3(3)
        >>> expected = actual
        >>> pp.testing.assert_lietensor_close(actual, expected)

    '''
    diff = (actual.Inv() @ expected).Log().tensor()
    zero = torch.zeros_like(diff)
    torch.testing.assert_close(diff, zero, *args, **kwargs)
