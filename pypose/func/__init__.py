import torch
from typing import Callable, Union, Tuple, Optional

from pypose.lietensor.lietensor import retain_ltype


def jacrev(func: Callable, argnums: Union[int, Tuple[int]] = 0, *, has_aux=False,
           chunk_size: Optional[int] = None,
           _preallocate_and_copy=False) -> Callable:
    """
    This function provides the exact same functionality as `torch.func.jacrev`, except
    that it allows LieTensor to be used as input.
    """
    jac_func = torch.func.jacrev(func, argnums, has_aux=has_aux, chunk_size=chunk_size,
        _preallocate_and_copy=_preallocate_and_copy)
    @retain_ltype()
    def wrapper_fn(*args, **kwargs):
        return jac_func(*args, **kwargs)
    return wrapper_fn
