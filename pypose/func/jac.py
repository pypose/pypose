import torch
from .. import retain_ltype
from typing import Callable, Union, Tuple, Optional


def jacrev(func: Callable, argnums: Union[int, Tuple[int]] = 0, *, has_aux=False,
           chunk_size: Optional[int] = None,
           _preallocate_and_copy=False):
    r"""
    This function provides the exact same functionality as `torch.func.jacrev()
    <https://pytorch.org/docs/stable/generated/torch.func.jacrev.html#torch.func.jacrev>`_,
    except that it allows LieTensor to be used as input when calculating the jacobian.

    Args:
        func (function): A Python function that takes one or more arguments,
            one of which must be a Tensor, and returns one or more Tensors
        argnums (int or Tuple[int]): Optional, integer or tuple of integers,
            saying which arguments to get the Jacobian with respect to.
            Default: 0.
        has_aux (bool): Flag indicating that ``func`` returns a
            ``(output, aux)`` tuple where the first element is the output of
            the function to be differentiated and the second element is
            auxiliary objects that will not be differentiated.
            Default: False.
        chunk_size (None or int): If None (default), use the maximum chunk size
            (equivalent to doing a single vmap over vjp to compute the jacobian).
            If 1, then compute the jacobian row-by-row with a for-loop.
            If not None, then compute the jacobian :attr:`chunk_size` rows at a time
            (equivalent to doing multiple vmap over vjp). If you run into memory issues
            computing the jacobian, please try to specify a non-None chunk_size.

    Returns:
        Returns a function that takes in the same inputs as ``func`` and
        returns the Jacobian of ``func`` with respect to the arg(s) at
        ``argnums``. If ``has_aux is True``, then the returned function
        instead returns a ``(jacobian, aux)`` tuple where ``jacobian``
        is the Jacobian and ``aux`` is auxiliary objects returned by ``func``.

    A basic usage with our LieTensor type would be the transformation function.

        >>> import pypose as pp
        >>> import torch
        >>> def func(pose, points):
        ...     return pose @ points
        >>> pose = pp.randn_SE3(1)
        >>> points = torch.randn(1, 3)
        >>> jacobian = pp.func.jacrev(func)(pose, points)
        >>> jacobian
        tensor([[[[ 1.0000,  0.0000,  0.0000,  0.0000,  1.5874, -0.2061,  0.0000]],
                [[ 0.0000,  1.0000,  0.0000, -1.5874,  0.0000, -1.4273,  0.0000]],
                [[ 0.0000,  0.0000,  1.0000,  0.2061,  1.4273,  0.0000,  0.0000]]]])
    """
    jac_func = torch.func.jacrev(func, argnums, has_aux=has_aux, chunk_size=chunk_size,
        _preallocate_and_copy=_preallocate_and_copy)
    @retain_ltype()
    def wrapper_fn(*args, **kwargs):
        return jac_func(*args, **kwargs)
    return wrapper_fn
